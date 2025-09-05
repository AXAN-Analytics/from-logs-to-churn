# syntax=docker/dockerfile:1.7


## INSTALLING PYTHON3.11 ON DEBIAN 12 : STABLE AND LIGHTWEIGHT
FROM python:3.11-slim-bookworm 

# ---- security & ergonomics ----


### WON'T CREATE __PYCACHE__ FOLDER (CONTAINING .PYC COMPILED PYTHON CODE) TO KEEP REPO CLEAN
### SEND LOGS TO REDIRECTION : NO INTERMEDIARY STDOUT
### AVOID KEEPING ARCHIVE IN A CACHE DIRECTORY ~/.cache/pip TO KEEP IMAGE LIGHTWEIGHT
### CACHE IS REDIRECT TO /tmp/.cache/ 
### DEFAULT TIMEZONE UTC+0 (LISBON TIME) FOR EVERY PROGRAM
### HOME DIRECTORY FOR NON-ROOT USER CREATED SET AT /home/freelance_user
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    TZ=UTC \
    HOME=/home/freelance_user

# Minimal system deps for SSL + stable DNS


### UPDATES APT METADTA, INSTALLS :

    ### APT : Advanced Package Tool
    ### CA : Certificate Authorit
    ### OS DEPENCIES ARE INSTALELD AND UPDATYED THROUGH THE PADCKAGE MAANGER (APT)
    ### HERE INSTALLING CA-CERTIFCATES -> SSL CERTICIFACATES NEEDED FOR HTTPS (FILES APPS CONSULKT WHEN MAKING SECURE CONNECTIONS)
    ### TZDATA : DATA TIME REFERENCE
    ### DO NOT INSTALL RECOMMENDED PACKAGES TO KEEP IMAGE LIGHT, DO NOT INSTALL RECOMMANATIONS AS WELL

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

# ---- app user (non-root) ----

### BUILD RIME ARGS
### UID/GID
### APP RUNS AS NONROOT
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g ${APP_GID} freelance_group \
 && useradd -r -u ${APP_UID} -g ${APP_GID} -d /home/freelance_user -m freelance_user


### WOKRING DIRECTORY IS /app
WORKDIR /app

# ---- Python deps first (better layer caching) ----

### COPY REQUIREMENTS.TXT INTO FOLDER APP
### RUNS PIP TO INSTALL ENVIRONMENT, PUT CACHE INSIDE /root/.cache/pip DURING OPERATION BUT DOES NOT KEEP IT
COPY requirements.txt /app/requirements.txt
# TEMP DIAGNOSTIC (more logs)
# Show verbose logs; on failure, print pip environment details
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip && \
    python -m pip install -v --no-cache-dir -r /app/requirements.txt || \
    (echo '--- pip debug below ---' && python -m pip debug -v && exit 1)

# ---- App code ----
# Copy your repo into the image; adjust the path if your project root differs.
# Exclude heavy/dev files via .dockerignore below.
COPY . /app

# ---- Minimal, idempotent ETL entrypoint (safe default CMD) ----
# Reads CONFIG_PATH (default: /app/config/postgres.json), connects with SSL,
# creates tables if missing, upserts a couple of sample rows, and prints event_mix.
# Keep this tiny and dependency-free.
RUN <<'PY' bash -lc 'cat > /main.py'
import json, os, sys, time
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text

def log(level, msg, **kw):
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "msg": msg}
    record.update(kw)
    print(json.dumps(record), flush=True)

cfg_path = os.getenv("CONFIG_PATH", "/app/config/postgres.json")
if not os.path.exists(cfg_path):
    log("error", "CONFIG_PATH missing", CONFIG_PATH=cfg_path)
    sys.exit(2)

try:
    cfg = json.load(open(cfg_path, "r"))
    host, port, db, user, pwd = cfg["host"], cfg["port"], cfg["database"], cfg["user"], cfg["password"]
    sslmode = cfg.get("sslmode", "require")
except Exception as e:
    log("error", "failed to parse config", error=str(e))
    sys.exit(2)

url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode={sslmode}"
engine = create_engine(url, pool_pre_ping=True)

ddl = """
CREATE TABLE IF NOT EXISTS users (
  user_id BIGSERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS events (
  event_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(user_id),
  event_type TEXT NOT NULL,
  event_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Unique per user+type per hour to keep this sample idempotent:
CREATE UNIQUE INDEX IF NOT EXISTS ux_events_user_type_hour
ON events (user_id, event_type, date_trunc('hour', event_ts));
"""

sample_upserts = """
-- Upsert two users
INSERT INTO users (email) VALUES
  ('alice@example.com'),
  ('bob@example.com')
ON CONFLICT (email) DO NOTHING;

-- Insert one event per user per hour; no dupes thanks to unique index
INSERT INTO events (user_id, event_type, event_ts)
SELECT u.user_id, e.event_type, date_trunc('hour', now())
FROM (VALUES
  ('alice@example.com','signup'),
  ('bob@example.com','login')
) AS e(email, event_type)
JOIN users u ON u.email = e.email
ON CONFLICT (user_id, event_type, date_trunc('hour', event_ts)) DO NOTHING;
"""

query_mix = """
SELECT event_type, COUNT(*) AS n_events
FROM events
GROUP BY event_type
ORDER BY n_events DESC, event_type ASC;
"""

try:
    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        conn.execute(text(sample_upserts))
        df = pd.read_sql(text(query_mix), conn)
        log("info", "event_mix", rows=len(df))
        # Print a human-friendly table too (what you’ll look for in logs)
        print(df.to_string(index=False))
    log("info", "etl_done")
    sys.exit(0)
except Exception as e:
    log("error", "etl_failed", error=str(e))
    sys.exit(1)
PY

# ---- Runtime security defaults ----

### shifting to non root user created earlier
USER ${APP_UID}:${APP_GID}

###defining CONFIG_PATH as variabke for
ENV CONFIG_PATH=/app/config/postgres.json

# Default command runs the small end-to-end ETL above.
CMD ["python", "/main.py"]
