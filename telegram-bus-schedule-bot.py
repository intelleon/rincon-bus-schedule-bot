"""
Telegram Bus Schedule Bot

Features
- Main menu with 3 options:
  1) Select destination
  2) Select a bus stop
  3) Show bus schedule
- Flow: destination → stop → schedule
- "Show bus schedule" also works from the main menu if destination/stop are already chosen.
- Back/Home navigation.

Setup
1) pip install python-telegram-bot==21.4
2) Set env var TELEGRAM_BOT_TOKEN with your bot token.
3) python telegram-bus-schedule-bot.py

Tested with python-telegram-bot v21.x (async API).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    AIORateLimiter,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
)

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Conversation states ---
SELECTING_DESTINATION, SELECTING_STOP = range(2)

# --- Demo data ---
# Map of destinations → stops → (demo) schedule generator parameters
DESTINATIONS: Dict[str, Dict[str, Dict[str, int]]] = {
    "Downtown": {
        "Main St & 3rd": {"start_min": 6 * 60, "every_min": 15},
        "Central Station": {"start_min": 5 * 60 + 30, "every_min": 20},
        "Museum Avenue": {"start_min": 7 * 60, "every_min": 12},
    },
    "Airport": {
        "Terminal A": {"start_min": 4 * 60 + 45, "every_min": 30},
        "Long Term Parking": {"start_min": 5 * 60, "every_min": 25},
    },
    "University": {
        "Library": {"start_min": 6 * 60 + 10, "every_min": 10},
        "Sports Complex": {"start_min": 6 * 60 + 5, "every_min": 15},
        "West Gate": {"start_min": 5 * 60 + 40, "every_min": 20},
    },
}

# --- Helpers ---

def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("1️⃣ Select destination", callback_data="MENU:SELECT_DEST")],
            [InlineKeyboardButton("2️⃣ Select a bus stop", callback_data="MENU:SELECT_STOP")],
            [InlineKeyboardButton("3️⃣ Show bus schedule", callback_data="MENU:SHOW_SCHEDULE")],
        ]
    )


def back_home_keyboard(include_back: bool = True) -> InlineKeyboardMarkup:
    rows = []
    if include_back:
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data="NAV:BACK")])
    rows.append([InlineKeyboardButton("🏠 Home", callback_data="NAV:HOME")])
    return InlineKeyboardMarkup(rows)


def destinations_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for dest in DESTINATIONS.keys():
        rows.append([InlineKeyboardButton(dest, callback_data=f"DEST:{dest}")])
    rows.append([InlineKeyboardButton("🏠 Home", callback_data="NAV:HOME")])
    return InlineKeyboardMarkup(rows)


def stops_keyboard(destination: str) -> InlineKeyboardMarkup:
    rows = []
    for stop in DESTINATIONS[destination].keys():
        rows.append([InlineKeyboardButton(stop, callback_data=f"STOP:{destination}:{stop}")])
    rows.append([
        InlineKeyboardButton("⬅️ Back", callback_data="NAV:BACK"),
        InlineKeyboardButton("🏠 Home", callback_data="NAV:HOME"),
    ])
    return InlineKeyboardMarkup(rows)


def generate_schedule_times(start_minutes: int, every_minutes: int, count: int = 12) -> List[str]:
    """Generate `count` upcoming times today based on a start-of-day offset and frequency."""
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    first = midnight + timedelta(minutes=start_minutes)

    # Move forward until next time in the future (or now)
    while first < now:
        first += timedelta(minutes=every_minutes)

    times = [first + i * timedelta(minutes=every_minutes) for i in range(count)]
    return [t.strftime("%H:%M") for t in times]


def format_selection(user_data: dict) -> str:
    dest = user_data.get("destination")
    stop = user_data.get("stop")
    parts = []
    parts.append(f"Destination: {dest if dest else '—'}")
    parts.append(f"Stop: {stop if stop else '—'}")
    return "\n".join(parts)


async def send_home(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🚌 *Bus Schedule Bot*\n\n"
        "Use the menu below to:\n"
        "1) Select destination\n"
        "2) Select a bus stop\n"
        "3) Show bus schedule"
    )
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(
            text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")


# --- Command handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.setdefault("destination", None)
    context.user_data.setdefault("stop", None)
    await send_home(update, context)
    return ConversationHandler.END


# --- Menu flow ---
async def on_menu_select(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    action = query.data.split(":", 1)[1]
    if action == "SELECT_DEST":
        await query.edit_message_text(
            text="Select a destination:", reply_markup=destinations_keyboard()
        )
        return SELECTING_DESTINATION
    elif action == "SELECT_STOP":
        dest = context.user_data.get("destination")
        if not dest:
            await query.edit_message_text(
                text=(
                    "Please select a *destination* first.",
                ),
                parse_mode="Markdown",
                reply_markup=destinations_keyboard(),
            )
            return SELECTING_DESTINATION
        await query.edit_message_text(
            text=f"Destination: *{dest}*\nChoose a stop:",
            parse_mode="Markdown",
            reply_markup=stops_keyboard(dest),
        )
        return SELECTING_STOP
    elif action == "SHOW_SCHEDULE":
        await show_schedule(update, context)
        return ConversationHandler.END
    else:
        await send_home(update, context)
        return ConversationHandler.END


async def on_destination(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    _, dest = query.data.split(":", 1)
    context.user_data["destination"] = dest
    context.user_data["stop"] = None  # reset stop when destination changes

    await query.edit_message_text(
        text=f"Destination: *{dest}*\nChoose a stop:",
        parse_mode="Markdown",
        reply_markup=stops_keyboard(dest),
    )
    return SELECTING_STOP


async def on_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    _, dest, stop = query.data.split(":", 2)
    context.user_data["destination"] = dest
    context.user_data["stop"] = stop

    # Immediately show the schedule after full selection
    await show_schedule(update, context, edit=True)
    return ConversationHandler.END


async def nav(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    _, where = query.data.split(":", 1)
    if where == "HOME":
        await send_home(update, context)
        return ConversationHandler.END
    elif where == "BACK":
        dest = context.user_data.get("destination")
        if dest:
            # Go back to the stops list for chosen destination
            await query.edit_message_text(
                text=f"Destination: *{dest}*\nChoose a stop:",
                parse_mode="Markdown",
                reply_markup=stops_keyboard(dest),
            )
            return SELECTING_STOP
        else:
            # Back goes to destinations list if no destination chosen yet
            await query.edit_message_text(
                text="Select a destination:", reply_markup=destinations_keyboard()
            )
            return SELECTING_DESTINATION
    return ConversationHandler.END


# --- Schedule ---
async def show_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE, edit: bool = False) -> None:
    """Show schedule if selection is complete; otherwise prompt to pick destination/stop."""
    dest = context.user_data.get("destination")
    stop = context.user_data.get("stop")

    if not dest or not stop:
        text = (
            "You haven't selected a full route yet.\n\n"
            "Choose *Select destination* and then *Select a bus stop*.")
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=text, parse_mode="Markdown", reply_markup=main_menu_keyboard()
            )
        else:
            await update.effective_message.reply_text(
                text, parse_mode="Markdown", reply_markup=main_menu_keyboard()
            )
        return

    params = DESTINATIONS.get(dest, {}).get(stop)
    if not params:
        msg = "No schedule data available for this selection."
        if edit and update.callback_query:
            await update.callback_query.edit_message_text(msg, reply_markup=back_home_keyboard())
        else:
            await update.effective_message.reply_text(msg, reply_markup=back_home_keyboard())
        return

    times = generate_schedule_times(params["start_min"], params["every_min"], count=10)
    header = f"🚌 *Next buses*\nDestination: *{dest}*\nStop: *{stop}*\n"
    body = "\n".join(f"• {t}" for t in times)
    footer = "\n\nTip: Use Home to change destination or stop."
    text = header + "\n" + body + footer

    if update.callback_query:
        await update.callback_query.edit_message_text(
            text=text, parse_mode="Markdown", reply_markup=back_home_keyboard(include_back=True)
        )
    else:
        await update.effective_message.reply_text(
            text, parse_mode="Markdown", reply_markup=back_home_keyboard(include_back=True)
        )


# --- Boilerplate ---
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Commands:\n"
        "/start – show main menu\n"
        "/help – this help\n\n"
        "Use the buttons to select destination → stop → schedule."
    )


def build_application() -> Application:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN env var. Set it to your BotFather token."
        )

    app = (
        Application.builder()
        .token(token)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # Conversation handlers for selection steps
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    # Menu actions
    app.add_handler(CallbackQueryHandler(on_menu_select, pattern=r"^MENU:"))

    # Destinations & stops
    app.add_handler(CallbackQueryHandler(on_destination, pattern=r"^DEST:"))
    app.add_handler(CallbackQueryHandler(on_stop, pattern=r"^STOP:"))

    # Navigation
    app.add_handler(CallbackQueryHandler(nav, pattern=r"^NAV:"))

    return app


# --- Simplified entrypoint using run_polling (avoids dropped updates) ---
def main() -> None:
    app = build_application()
    # Ensure webhook is disabled implicitly by run_polling
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
