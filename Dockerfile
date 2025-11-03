# Slim Python
FROM python:3.12-slim

# OS deps (certs + timezone)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
  && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Madrid
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Configure app env (SQLite path lives on a mounted volume)
ENV DB_PATH=/data/bus_reports.db
ENV DEFAULT_LINEA=19
ENV GEOFENCE_METERS=500
ENV PYTHONUNBUFFERED=1

# Start
CMD [ "bash", "-lc", "mkdir -p /data && python main.py" ]
