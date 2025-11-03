import asyncio
import math
import os
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from zoneinfo import ZoneInfo
from datetime import datetime

import aiosqlite
import aiohttp
from dotenv import load_dotenv
from telegram import (
    Update,
    KeyboardButton,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# --- Config ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_IDS = set(int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit())
GEOFENCE_METERS = int(os.getenv("GEOFENCE_METERS", "500") or 500)  # default 500 m
DB_PATH = os.getenv("DB_PATH", "bus_reports.db")
API_STOPS_URL = os.getenv("STOPS_API_URL", "http://api.ctan.es/v1/Consorcios/4/lineas/19/paradas")
API_SCHEDULE_URL = os.getenv(
    "SCHEDULE_API_URL",
    "http://api.ctan.es/v1/Consorcios/4/horarios_lineas?dia={dia}&frecuencia=&lang=ES&linea={linea}&mes={mes}",
)
DEFAULT_LINEA = os.getenv("DEFAULT_LINEA", "19")
PAGE_SIZE = 8  # stops per page in inline keyboards
# --- Temporary UI toggles (do not delete logic; just hide UI) ---
SHOW_LOCATION_UI = False   # hide "Compartir ubicación" buttons/prompts
SHOW_BLOQUES_UI  = False   # hide "Ver horarios por bloques" buttons/prompts


# --- Session state ---
@dataclass
class UserSession:
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    last_loc_at: Optional[float] = None
    stop_id: Optional[int] = None
    stop_name: Optional[str] = None
    route: Optional[str] = None            # linea
    route: str = DEFAULT_LINEA
    destination: Optional[str] = None      # "malaga" | "rincon"
    sentido: Optional[str] = None          # "1" | "2"
    page: int = 0

sessions: Dict[int, UserSession] = {}

# --- Utils ---
def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def today_madrid():
    tz = ZoneInfo("Europe/Madrid")
    now = datetime.now(tz)
    return now.day, now.month, now

# --- DB ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS bus_stops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ext_stop_id TEXT NOT NULL,
                line_id TEXT NOT NULL,
                nucleus TEXT,
                zone TEXT,
                name TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                sentido TEXT NOT NULL,
                orden INTEGER,
                UNIQUE(ext_stop_id, line_id, sentido)
            );

            CREATE TABLE IF NOT EXISTS bus_status_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT,
                route TEXT NOT NULL,
                stop_id INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('on_time','late')),
                created_at INTEGER NOT NULL,
                FOREIGN KEY (stop_id) REFERENCES bus_stops(id)
            );
            """
        )
        await db.commit()

async def fetch_and_upsert_stops() -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(API_STOPS_URL, timeout=20) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
        except Exception as e:
            print(f"[WARN] Failed to fetch stops: {e}")
            return 0

        paradas = data.get("paradas") if isinstance(data, dict) else data
        if not paradas:
            print("[WARN] Empty or malformed stops payload")
            return 0

        inserted = 0
        for p in paradas:
            try:
                await db.execute(
                    """
                    INSERT OR IGNORE INTO bus_stops(
                        ext_stop_id, line_id, nucleus, zone, name, lat, lon, sentido, orden
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(p.get("idParada")),
                        str(p.get("idLinea")),
                        str(p.get("idNucleo")),
                        str(p.get("idZona")),
                        p.get("nombre"),
                        float(p.get("latitud")),
                        float(p.get("longitud")),
                        str(p.get("sentido")),
                        int(p.get("orden", 0) or 0),
                    ),
                )
                inserted += 1
            except Exception as ex:
                print(f"[WARN] Skipped stop: {ex}")
        await db.commit()
        print(f"[INFO] Upserted {inserted} stops")
        return inserted


# --- Persistent reply keyboard (bottom bar) ---
def main_reply_keyboard():
    rows = [[KeyboardButton("🏁 Menú")]]
    if SHOW_LOCATION_UI:
        rows.append([KeyboardButton(text="📍 Compartir ubicación", request_location=True)])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=False)




async def get_stop_with_ext(stop_id: int):
    """Return (id, ext_stop_id, name, lat, lon, line_id) for internal id."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, ext_stop_id, name, lat, lon, line_id FROM bus_stops WHERE id=?",
            (stop_id,),
        )
        return await cur.fetchone()



# --- Tiempo de paso (parada) ---

try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None

# --- Tiempo de paso (parada) with robust SSL handling ---
import ssl
try:
    import certifi
    _CERT_PATH = certifi.where()
except Exception:
    certifi = None
    _CERT_PATH = None

async def fetch_realtime_for_stop(ext_stop_id: str) -> str:
    """
    Fetch https://ctmam.es/tiempo-paso-parada/?parada={ext_stop_id}
    Return a clean, readable plain-text summary:
    Each record shows line info and 'Tiempo estimado' on the next line,
    with no blank lines between records.
    """
    if not ext_stop_id:
        return "No se pudo obtener el identificador de la parada."

    url = f"https://ctmam.es/tiempo-paso-parada/?parada={ext_stop_id}"
    html = None
    ssl_ctx = None
    if _CERT_PATH:
        try:
            import ssl
            ssl_ctx = ssl.create_default_context(cafile=_CERT_PATH)
        except Exception:
            ssl_ctx = None

    # --- Fetch HTML safely ---
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20, ssl=ssl_ctx) as resp:
                resp.raise_for_status()
                html = await resp.text()
    except Exception as e:
        return f"No se pudo cargar la página oficial ({e})."

    try:
        if BeautifulSoup:
            soup = BeautifulSoup(html, "html.parser")
            lines = []

            for sel in [".list-group-item", ".card-body", "table tr", "ul li", "ol li", "p"]:
                for el in soup.select(sel):
                    txt = el.get_text(" ", strip=True)
                    if txt:
                        lines.append(txt)

            if not lines:
                lines = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

            # De-duplicate
            seen = set()
            dedup = []
            for l in lines:
                l = l.strip()
                if not l:
                    continue
                key = l.lower()
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(l)

            # Drop subset lines (e.g., plain "Tiempo estimado" when a fuller line exists)
            filtered = []
            for line in dedup:
                if any(line.lower() in d.lower() and line.lower() != d.lower() for d in dedup):
                    continue
                filtered.append(line)

            # Build clean output
            formatted_lines = []
            for line in filtered:
                low = line.lower()
                if "tiempo estimado" in low:
                    # insert line break before 'Tiempo estimado'
                    parts = line.split("Tiempo estimado")
                    if len(parts) == 2:
                        formatted_lines.append(f"{parts[0].strip()}\nTiempo estimado{parts[1]}")
                    else:
                        formatted_lines.append(line)
                elif line.lower().startswith("m-") or "salida" in low:
                    # if it doesn’t contain "Tiempo estimado" itself, keep as-is
                    formatted_lines.append(line)
                else:
                    # skip generic headings like "Próximas salidas"
                    continue

            if formatted_lines:
                return "⏱️ Tiempo de paso (web oficial)\n" + "\n".join(formatted_lines)
                #return "" + "\n".join(formatted_lines)

        # fallback: plain strip
        import re as _re
        text = _re.sub(r"<[^>]+>", " ", html or "")
        text = " ".join(text.split())
        return "⏱️ Tiempo de paso (web oficial)\n" + text[:600]

    except Exception:
        return "No se pudo interpretar la página de la parada."

# --- Stop queries ---
async def get_stops_by_sentido(sentido: str) -> List[Tuple[int, str, float, float, str, int]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, name, lat, lon, line_id, orden FROM bus_stops WHERE sentido=? ORDER BY orden ASC",
            (sentido,),
        )
        return await cur.fetchall()

async def get_stop_by_id(stop_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id, name, lat, lon, line_id FROM bus_stops WHERE id=?", (stop_id,))
        return await cur.fetchone()

async def get_stop_by_name(name: str, sentido: Optional[str] = None):
    """Return (id, name, lat, lon, line_id, sentido, orden) by stop name; prefers current sentido."""
    async with aiosqlite.connect(DB_PATH) as db:
        if sentido:
            cur = await db.execute(
                """
                SELECT id, name, lat, lon, line_id, sentido, orden
                FROM bus_stops
                WHERE LOWER(name) = LOWER(?) AND sentido = ?
                ORDER BY orden ASC
                LIMIT 1
                """,
                (name, sentido),
            )
            row = await cur.fetchone()
            if row:
                return row
        cur = await db.execute(
            """
            SELECT id, name, lat, lon, line_id, sentido, orden
            FROM bus_stops
            WHERE LOWER(name) = LOWER(?)
            ORDER BY orden ASC
            LIMIT 1
            """,
            (name,),
        )
        return await cur.fetchone()

# --- Markdown escape (clean) ---
MD_RESERVED = "*_`"

def md_escape(text: str) -> str:
    """Escape only the characters Telegram Markdown actually needs."""
    if not text:
        return ""
    return "".join("\\" + ch if ch in MD_RESERVED else ch for ch in text)


def dest_label_for_sentido(sess: UserSession, sentido: str) -> str:
    # If user already chose destination, trust that
    if getattr(sess, "destination", None) in ("malaga", "rincon"):
        return "Málaga" if sess.destination == "malaga" else "Rincón de la Victoria"
    # Fallback: sentido 2 -> Málaga (vuelta), sentido 1 -> Rincón (ida)
    return "Málaga" if str(sentido) == "2" else "Rincón de la Victoria"


# --- Geofence ---
async def find_nearest_stop(lat: float, lon: float):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id, name, lat, lon, line_id FROM bus_stops")
        rows = await cur.fetchall()
    best = None
    best_d = float("inf")
    for stop_id, name, s_lat, s_lon, line_id in rows:
        d = haversine_meters(lat, lon, s_lat, s_lon)
        if d < best_d:
            best_d = d
            best = (stop_id, name, s_lat, s_lon, d, line_id)
    return best

async def within_geofence(lat, lon, stop_lat=None, stop_lon=None):
    if stop_lat is not None and stop_lon is not None:
        d = haversine_meters(lat, lon, stop_lat, stop_lon)
        if d <= GEOFENCE_METERS:
            return {"distance": d}
        return None
    nearest = await find_nearest_stop(lat, lon)
    if not nearest:
        return None
    stop_id, name, s_lat, s_lon, dist_m, line_id = nearest
    if dist_m <= GEOFENCE_METERS:
        return {"stop_id": stop_id, "stop_name": name, "distance": dist_m, "line_id": line_id}
    return None

# --- Schedule (planificador por BLOQUES) ---
async def fetch_schedule_json(linea: str, dia: int, mes: int) -> Any:
    url = API_SCHEDULE_URL.format(dia=dia, mes=mes, linea=linea)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)
    except Exception as e:
        print(f"[WARN] Failed to fetch schedule: {e}")
        return None

def pick_direction_keys(destination: str):
    # Destino Málaga -> Vuelta (hacia Málaga)
    # Destino Rincón -> Ida (hacia Rincón)
    if destination == "malaga":
        return ("bloquesVuelta", "horarioVuelta")
    return ("bloquesIda", "horarioIda")

async def fetch_planificador_for_today(linea: str, dia: int, mes: int):
    payload = await fetch_schedule_json(linea, dia, mes)
    if not payload:
        return None
    planis = payload.get("planificadores") if isinstance(payload, dict) else None
    if not planis:
        return None
    return planis[0]

def parse_bloques(planificador: dict, destination: str):
    """Return (bloque_names_filtered, horario_rows, raw_bloques) for the chosen direction.
       Excludes bloques with tipo == '1' (e.g., 'Frecuencia')."""
    bloquesKey, horarioKey = pick_direction_keys(destination)
    raw_bloques = [b for b in planificador.get(bloquesKey, []) if isinstance(b, dict)]
    # filter out tipo == '1'
    filtered = [b.get("nombre") for b in raw_bloques if str(b.get("tipo", "0")) != "1"]
    horario_rows = planificador.get(horarioKey, []) or []
    return filtered, horario_rows, raw_bloques

def times_for_bloque_index(horario_rows: list, col_index: int) -> List[str]:
    """Extract column index from each row.horas (aligned to filtered bloques)."""
    out: List[str] = []
    for row in horario_rows:
        horas = row.get("horas") if isinstance(row, dict) else None
        if not isinstance(horas, list):
            continue
        if 0 <= col_index < len(horas):
            val = horas[col_index]
            if isinstance(val, str) and ":" in val and val.strip() != "--":
                out.append(val.strip())
    # De-dup + sort time
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    uniq.sort(key=lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]))
    return uniq

# (Debug) collect any times
def _collect_times_anywhere(node) -> List[str]:
    times: List[str] = []
    time_re = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
    def scan(x):
        nonlocal times
        if isinstance(x, dict):
            for v in x.values():
                scan(v)
        elif isinstance(x, list):
            for v in x:
                scan(v)
        elif isinstance(x, str):
            times.extend(time_re.findall(x))
    scan(node)
    seen, out = set(), []
    for t in times:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# --- Keyboards ---
def destination_keyboard():
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("🏙️ Málaga", callback_data="dest:malaga"),
            InlineKeyboardButton("🏖️ Rincón de la Victoria", callback_data="dest:rincon"),
        ]]
    )

def location_keyboard():
    return ReplyKeyboardMarkup(
        [[KeyboardButton(text="📍 Compartir ubicación", request_location=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

def stop_picker_keyboard(stops: List[Tuple[int, str, float, float, str, int]], page: int, sentido: str):
    total = len(stops)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_items = stops[start:end]

    rows = []
    for (sid, name, _lat, _lon, _line, _ord) in page_items:
        rows.append([InlineKeyboardButton(name, callback_data=f"stop:{sid}")])

    controls = []
    if page > 0:
        controls.append(InlineKeyboardButton("⬅️ Anterior", callback_data=f"stoppage:{sentido}:{page-1}"))
    if end < total:
        controls.append(InlineKeyboardButton("Siguiente ➡️", callback_data=f"stoppage:{sentido}:{page+1}"))
    if controls:
        rows.append(controls)

    other = "2" if sentido == "1" else "1"
    label_other = f"Cambiar a sentido { 'Málaga' if other == '2' else 'Rincón de la Victoria' }"
    rows.append([InlineKeyboardButton(label_other, callback_data=f"stoppage:{other}:0")])

    rows.append([InlineKeyboardButton("↩️ Volver", callback_data="back:after_dest")])
    return InlineKeyboardMarkup(rows)

# --- Handlers ---
async def send_start_menu(chat, context: ContextTypes.DEFAULT_TYPE):
    # Explicit instruction line + destination buttons
    await chat.send_message("Selecciona destino:", reply_markup=destination_keyboard())
    # (No extra “Tienes el menú siempre abajo 👇” line)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    sessions[user.id] = UserSession(route=DEFAULT_LINEA)
    await send_start_menu(update.message.chat, context)

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    sessions[user.id] = UserSession(route=DEFAULT_LINEA)
    await send_start_menu(update.message.chat, context)



async def menu_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Works in any state; just re-show the start menu
    await send_start_menu(update.message.chat, context)

async def unknown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_start_menu(update.message.chat, context)


async def select_destination(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession(route=DEFAULT_LINEA))

    try:
        _, choice = query.data.split(":", 1)
    except ValueError:
        await query.edit_message_text("Selección inválida. Usa 🏁 Menú.")
        return
    if choice not in ("malaga", "rincon"):
        await query.edit_message_text("Selección inválida. Usa 🏁 Menú.")
        return

    # Save destination & direction; keep default line
    sess.destination = choice
    sess.sentido = "2" if choice == "malaga" else "1"
    sess.page = 0
    sess.route = sess.route or DEFAULT_LINEA

    # Immediately show the stop list (no extra messages)
    stops = await get_stops_by_sentido(sess.sentido)
    if not stops:
        await query.edit_message_text("No hay paradas disponibles.")
        return

    # Human-readable destination label
    dest_human = "Málaga" if sess.destination == "malaga" else "Rincón de la Victoria"

    await query.edit_message_text(
        f"Paradas (sentido {dest_human}). Página 1:",
        reply_markup=stop_picker_keyboard(stops, 0, sess.sentido),
    )


# After-destination: ask for location
async def ask_location_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Comparte tu ubicación para detectar la parada más cercana.", reply_markup=location_keyboard())

# Stops list flow
async def stop_picker_init(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())

    if not sess.sentido:
        await query.edit_message_text("Primero elige destino con /start.")
        return

    stops = await get_stops_by_sentido(sess.sentido)
    if not stops:
        await query.edit_message_text("No hay paradas disponibles.")
        return

    sess.page = 0

    dest_human = dest_label_for_sentido(sess, sess.sentido)
    await query.edit_message_text(
        f"Paradas (sentido {dest_human}). Página {sess.page+1}:",
        reply_markup=stop_picker_keyboard(stops, sess.page, sess.sentido),
    )

async def stop_picker_page(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())

    try:
        _, sentido, page_str = query.data.split(":")
        page = int(page_str)
    except Exception:
        await query.edit_message_text("Navegación inválida. Usa /start.")
        return

    sess.sentido = sentido
    sess.page = page

    stops = await get_stops_by_sentido(sentido)
    dest_human = dest_label_for_sentido(sess, sentido)

    if not stops:
        await query.edit_message_text("No hay paradas disponibles.")
        return

    await query.edit_message_text(
        f"Paradas (sentido {dest_human}). Página {page+1}:",
        reply_markup=stop_picker_keyboard(stops, page, sentido),
    )

async def stop_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle picking a bus stop from the list.
    - Shows CTMAM realtime ('tiempo de paso') for that stop.
    - If user is within geofence, shows status buttons to report on-time/late.
    - No bloques hint and no location prompt when toggles hide them.
    """
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())

    # Parse callback data: "stop:<internal_id>"
    try:
        _, sid_str = query.data.split(":")
        sid = int(sid_str)
    except Exception:
        await query.edit_message_text("Selección inválida. Usa 🏁 Menú.")
        return

    # Need ext_stop_id to query CTMAM
    row = await get_stop_with_ext(sid)
    if not row:
        await query.edit_message_text("Parada no encontrada. Usa 🏁 Menú.")
        return

    stop_id, ext_stop_id, stop_name, s_lat, s_lon, line_id = row

    # Update session
    sess.stop_id = stop_id
    sess.stop_name = stop_name
    # Ensure we always have a default line
    sess.route = (line_id or sess.route or DEFAULT_LINEA)

    # Fetch official "tiempo de paso" page for this stop (plain text, no markdown)
    realtime_block_raw = await fetch_realtime_for_stop(ext_stop_id)

    # Destination human name
    dest_human = "Málaga" if sess.destination == "malaga" else "Rincón de la Victoria"

    # If user already shared location, check proximity (no prompts when UI is hidden)
    can_report = False
    distance_line = ""
    if sess.last_lat and sess.last_lon:
        try:
            within = await within_geofence(sess.last_lat, sess.last_lon, s_lat, s_lon)
        except Exception:
            within = None
        if within:
            can_report = True
            distance_line = f" (a {int(within['distance'])} m)"
        else:
            # Keep a subtle note; no prompting to share location
            distance_line = f" (fuera de {GEOFENCE_METERS} m)"

    # Escape ALL variable chunks for Markdown
    dest_md = md_escape(dest_human)
    stop_md = md_escape(stop_name)
    route_md = md_escape(str(sess.route))
    dist_md = md_escape(distance_line)
    realtime_md = md_escape(realtime_block_raw)

    # Build the message (no bloques note, no location prompt)
    base_text = (
        f"Destino: *{dest_md}*\n"
        f"Parada seleccionada: *{stop_md}*{dist_md}\n"
        f"{realtime_md}"
    )


    # Build inline keyboard
    rows = []
    if can_report:
        rows.append([
            InlineKeyboardButton("✅ Llegó a tiempo", callback_data="on_time"),
            InlineKeyboardButton("⛔ Retrasado", callback_data="late"),
        ])
    # Optionally include bloques button if feature flag is on
    if SHOW_BLOQUES_UI:
        rows.append([InlineKeyboardButton("🧩 Ver horarios por *bloques*", callback_data="bloques:init")])
    rows.append([InlineKeyboardButton("↩️ Volver", callback_data="back:after_dest")])

    try:
        await query.edit_message_text(
            base_text + ("\n\nIndica el estado:" if can_report else ""),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(rows),
        )
    except Exception as ex:
        # As a last resort, send without markdown if any formatting error occurs
        fallback_text = (
            f"Destino: {dest_human}\n"
            f"Parada seleccionada: {stop_name}{distance_line}\n"
            f"Línea: {sess.route}\n"
            f"{realtime_block_raw}"
        )
        await query.edit_message_text(
            fallback_text + ("\n\nIndica el estado:" if can_report else ""),
            reply_markup=InlineKeyboardMarkup(rows),
        )

async def back_after_destination(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return to the post-destination menu.
    Respects UI toggles: SHOW_LOCATION_UI and SHOW_BLOQUES_UI.
    No header text shown if there's only one option.
    """
    query = update.callback_query
    await query.answer()

    # Build inline menu conditionally
    rows = []

    if SHOW_LOCATION_UI:
        rows.append([InlineKeyboardButton("📍 Compartir ubicación", callback_data="ask_location")])

    rows.append([InlineKeyboardButton("🧭 Elegir parada de la lista", callback_data="stoppage:init")])

    if SHOW_BLOQUES_UI:
        rows.append([InlineKeyboardButton("🧩 Ver horarios por *bloques*", callback_data="bloques:init")])

    try:
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(rows))
    except Exception:
        # fallback for Telegram API edge cases (e.g., message type)
        await query.edit_message_text(
            text="\u2063",  # invisible char so Telegram accepts edit
            reply_markup=InlineKeyboardMarkup(rows),
        )

# Location flow (detect nearest stop & allow reporting)
async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())

    if not sess.destination:
        await update.message.reply_text("Primero elige tu destino:", reply_markup=destination_keyboard())
        return

    loc = update.message.location
    sess.last_lat = loc.latitude
    sess.last_lon = loc.longitude
    sess.last_loc_at = time.time()

    fence = await within_geofence(loc.latitude, loc.longitude)
    if not fence:
        await update.message.reply_text(
            f"❌ No estás dentro de {GEOFENCE_METERS} m de una parada conocida. Acércate e inténtalo de nuevo."
        )
        return

    sess.stop_id = fence.get("stop_id")
    sess.stop_name = fence.get("stop_name")
    sess.route = fence.get("line_id") or sess.route or DEFAULT_LINEA

    dest_human = "Málaga" if sess.destination == "malaga" else "Rincón de la Victoria"
    await update.message.reply_text(
        (
            f"Destino: *{dest_human}*\n"
            f"📍 Parada detectada: *{sess.stop_name}* (a {int(fence['distance'])} m)\n"
            f"Línea detectada: *{sess.route}*\n"
            "ℹ️ El horario oficial se muestra por *bloques*. Usa el botón de abajo para consultarlo."
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Llegó a tiempo", callback_data="on_time"),
                 InlineKeyboardButton("⛔ Retrasado", callback_data="late")],
                [InlineKeyboardButton("🧩 Ver horarios por *bloques*", callback_data="bloques:init")],
            ]
        ),
    )

# Route correction by text (no per-stop times)
async def handle_route_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())
    text = (update.message.text or "").strip()
    if not text:
        return

    # If the user hasn't picked destination yet, show start menu instead of parsing text
    if not sess.destination:
        await send_start_menu(update.message.chat, context)
        return

    # Otherwise keep the old behavior (manual route override)
    if not sess.stop_id:
        await update.message.reply_text("Por favor, selecciona destino y parada primero con /start.")
        return

    sess.route = text
    await update.message.reply_text(
        f"Línea actualizada a: *{md_escape(sess.route)}*\n"
        "ℹ️ Recuerda: los horarios se muestran por *bloques*.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🧩 Ver horarios por *bloques*", callback_data="bloques:init")]]),
    )


# Submit status
async def submit_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user = update.effective_user
    await query.answer()
    status_choice = query.data
    sess = sessions.get(user.id)

    if not sess or not (sess.stop_id and sess.route and sess.last_lat and sess.last_lon):
        await query.edit_message_text("Faltan datos. Usa /start e inténtalo de nuevo.")
        return

    row = await get_stop_by_id(sess.stop_id)
    if row:
        _sid, _name, s_lat, s_lon, _line = row
        within = await within_geofence(sess.last_lat, sess.last_lon, s_lat, s_lon)
        if not within:
            await query.edit_message_text(
                f"❌ Ya no estás dentro de {GEOFENCE_METERS} m de la parada seleccionada. Comparte ubicación otra vez."
            )
            return

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO bus_status_records(user_id, username, route, stop_id, status, created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (
                user.id,
                user.username or user.full_name,
                str(sess.route),
                sess.stop_id,
                status_choice,
                int(time.time()),
            ),
        )
        await db.commit()

    emoji, txt = ("✅", "Llegó a tiempo") if status_choice == "on_time" else ("⛔", "Retrasado")
    dest_human = "Málaga" if sess.destination == "malaga" else "Rincón de la Victoria"
    await query.edit_message_text(
        f"{emoji} Registrado: *{txt}*\n"
        f"Destino: *{dest_human}*\n"
        f"Parada: *{sess.stop_name}*\n"
        f"Línea: *{sess.route}*",
        parse_mode=ParseMode.MARKDOWN,
    )

# BLOQUES flow
async def bloques_init(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())
    if not sess.destination:
        await query.edit_message_text("Primero elige destino con /start.")
        return

    dia, mes, _ = today_madrid()
    plani = await fetch_planificador_for_today(sess.route or DEFAULT_LINEA, dia, mes)
    if not plani:
        await query.edit_message_text("No se pudo obtener el planificador de hoy.")
        return

    bloque_names, _rows, raw_bloques = parse_bloques(plani, sess.destination)
    if not bloque_names:
        await query.edit_message_text("No hay bloques disponibles.")
        return

    # Buttons one per bloque (you can paginate if needed)
    rows = [[InlineKeyboardButton(nom, callback_data=f"bloque:{i}") ] for i, nom in enumerate(bloque_names)]
    rows.append([InlineKeyboardButton("↩️ Volver", callback_data="back:after_dest")])
    await query.edit_message_text("Elige un bloque para ver horarios:", reply_markup=InlineKeyboardMarkup(rows))

async def bloque_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())
    if not sess.destination:
        await query.edit_message_text("Primero elige destino con /start.")
        return

    try:
        _, idx_s = query.data.split(":")
        idx = int(idx_s)
    except:
        await query.edit_message_text("Selección inválida de bloque.")
        return

    dia, mes, now = today_madrid()
    plani = await fetch_planificador_for_today(sess.route or DEFAULT_LINEA, dia, mes)
    if not plani:
        await query.edit_message_text("No se pudo obtener el planificador de hoy.")
        return

    bloque_names, horario_rows, raw_bloques = parse_bloques(plani, sess.destination)
    if not bloque_names or idx < 0 or idx >= len(bloque_names):
        await query.edit_message_text("Bloque no encontrado.")
        return

    bloque_name = bloque_names[idx]
    times = times_for_bloque_index(horario_rows, idx)

    dest_human = "Málaga" if sess.destination == "malaga" else "Rincón de la Victoria"

    # 🔎 Look up bloque coordinates from bus_stops by name (names coincide)
    stop_row = await get_stop_by_name(bloque_name, sess.sentido)
    coords_line = ""
    if stop_row:
        _sid, _name, s_lat, s_lon, _line, _sent, _ord = stop_row
        coords_line = f"\n📍 Coordenadas bloque: `{s_lat:.6f}, {s_lon:.6f}`"
        # If user shared location, show distance
        if sess.last_lat and sess.last_lon:
            dist = haversine_meters(sess.last_lat, sess.last_lon, s_lat, s_lon)
            coords_line += f" — a {int(dist)} m de tu ubicación"

    if times:
        now_mins = now.hour * 60 + now.minute
        fut = [t for t in times if (int(t[:2]) * 60 + int(t[3:])) >= now_mins] or times
        texto = (
            f"🕘 Salidas hoy hacia *{dest_human}* — bloque *{bloque_name}*:\n"
            + " • ".join(fut[:12])
            + coords_line
        )
    else:
        texto = f"🕘 No hay horarios publicados para el bloque *{bloque_name}* hoy." + coords_line

    await query.edit_message_text(texto, parse_mode=ParseMode.MARKDOWN)

# Commands: stats, sync, help, stops, bloques, horario, debugschedule
async def my_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT r.id, s.name, r.route, r.status, r.created_at
            FROM bus_status_records r
            JOIN bus_stops s ON s.id = r.stop_id
            WHERE r.user_id = ?
            ORDER BY r.id DESC
            LIMIT 10
            """,
            (user.id,),
        )
        rows = await cur.fetchall()

    if not rows:
        await update.message.reply_text("No tienes registros todavía. Usa /start para comenzar.")
        return

    lines = ["🧾 Tus últimos 10 reportes:"]
    for rid, stop_name, route, status, ts in rows:
        estado_txt = "A tiempo" if status == "on_time" else "Retrasado"
        lines.append(f"• #{rid} — {stop_name} — Línea {route} — {estado_txt}")
    await update.message.reply_text("\n".join(lines))

async def sync_stops(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else None
    if ADMIN_IDS and user_id not in ADMIN_IDS:
        await update.message.reply_text("Solo administradores.")
        return
    n = await fetch_and_upsert_stops()
    await update.message.reply_text(f"🔄 Sincronizadas {n} paradas desde la API.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — iniciar y elegir destino\n"
        "/menu — volver a elegir destino\n"
        "/stops — abrir la lista de paradas (para reportar)\n"
        "/bloques — ver horarios por *bloques*\n"
        "/mystats — ver tus últimos reportes\n"
        "/horario — ver horarios (bloques) para hoy\n"
        "/debugschedule — depurar JSON de horarios\n"
        "/syncstops — admin: actualizar paradas desde API\n"
        f"Radio de validación: {GEOFENCE_METERS} m\nAPI paradas: {API_STOPS_URL}\nAPI horarios: {API_SCHEDULE_URL}"
    )

async def stops_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())
    if not sess.destination:
        await update.message.reply_text("Primero elige tu destino:", reply_markup=destination_keyboard())
        return
    if not sess.sentido:
        sess.sentido = "2" if sess.destination == "malaga" else "1"
    stops = await get_stops_by_sentido(sess.sentido)
    if not stops:
        await update.message.reply_text("No hay paradas disponibles.")
        return
    sess.page = 0
    await update.message.reply_text(
        f"Paradas (sentido {sess.sentido}). Página {sess.page+1}:",
        reply_markup=stop_picker_keyboard(stops, sess.page, sess.sentido),
    )

async def bloques_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # convenience command to open bloques menu
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())
    if not sess.destination:
        await update.message.reply_text("Primero elige tu destino:", reply_markup=destination_keyboard())
        return
    dia, mes, _ = today_madrid()
    plani = await fetch_planificador_for_today(sess.route or DEFAULT_LINEA, dia, mes)
    if not plani:
        await update.message.reply_text("No se pudo obtener el planificador de hoy.")
        return
    bloque_names, _rows, _raw = parse_bloques(plani, sess.destination)
    if not bloque_names:
        await update.message.reply_text("No hay bloques disponibles.")
        return
    rows = [[InlineKeyboardButton(nom, callback_data=f"bloque:{i}") ] for i, nom in enumerate(bloque_names)]
    await update.message.reply_text("Elige un bloque para ver horarios:", reply_markup=InlineKeyboardMarkup(rows))

async def horario_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # convenience: open bloques menu (since times are per bloque)
    await bloques_cmd(update, context)

async def debugschedule_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    sess = sessions.setdefault(user.id, UserSession())
    dia, mes, _ = today_madrid()
    payload = await fetch_schedule_json(sess.route or DEFAULT_LINEA, dia, mes)
    if payload is None:
        await update.message.reply_text("No se pudo obtener el JSON de horarios.")
        return
    try:
        if isinstance(payload, dict):
            keys = list(payload.keys())
            planis = payload.get("planificadores")
            subkeys = []
            if isinstance(planis, list) and planis:
                p0 = planis[0]
                if isinstance(p0, dict):
                    subkeys = list(p0.keys())[:12]
            msg = f"Claves raíz: {keys}\nPlanificador[0] claves: {subkeys}"
        else:
            msg = f"Tipo raíz: {type(payload).__name__}"
        sample = _collect_times_anywhere(payload)[:20]
        if sample:
            msg += "\nMuestras de HH:MM: " + " ".join(sample)
        await update.message.reply_text(msg[:4000])
    except Exception:
        await update.message.reply_text("JSON de horarios recibido pero no se pudo resumir.")

# --- Entrypoint (Python 3.12-friendly) ---
def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set")

    asyncio.run(init_db())
    try:
        asyncio.run(fetch_and_upsert_stops())
    except Exception as e:
        print(f"[WARN] Initial stops sync failed: {e}")

    app: Application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("stops", stops_cmd))
    app.add_handler(CommandHandler("bloques", bloques_cmd))
    app.add_handler(CommandHandler("mystats", my_stats))
    app.add_handler(CommandHandler("horario", horario_cmd))
    app.add_handler(CommandHandler("debugschedule", debugschedule_cmd))
    app.add_handler(CommandHandler("syncstops", sync_stops))

    # Callback flows
    app.add_handler(CallbackQueryHandler(select_destination, pattern=r"^dest:(malaga|rincon)$"))
    app.add_handler(CallbackQueryHandler(ask_location_button, pattern=r"^ask_location$"))
    app.add_handler(CallbackQueryHandler(stop_picker_init, pattern=r"^stoppage:init$"))
    app.add_handler(CallbackQueryHandler(stop_picker_page, pattern=r"^stoppage:(1|2):\d+$"))
    app.add_handler(CallbackQueryHandler(stop_select, pattern=r"^stop:\d+$"))
    app.add_handler(CallbackQueryHandler(back_after_destination, pattern=r"^back:after_dest$"))
    app.add_handler(CallbackQueryHandler(bloques_init, pattern=r"^bloques:init$"))
    app.add_handler(CallbackQueryHandler(bloque_select, pattern=r"^bloque:\d+$"))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^\s*🏁\s*Menú\s*$"), menu_button_handler))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_route_text))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))


    # Status buttons
    app.add_handler(CallbackQueryHandler(submit_status, pattern=r"^(on_time|late)$"))

    print("✅ Bot is running… Press Ctrl+C to stop.")

    # Python 3.12: ensure a loop exists before run_polling()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app.run_polling()

if __name__ == "__main__":
    main()
