# pg_etl.py
from __future__ import annotations
import os
import io
import csv
import json
import pandas as pd
import pathlib
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DEFAULT_CFG = pathlib.Path.home() / ".config" / "OVH-cloud" / "from-logs-to-churn" / "postgres.json"

class DataBase:

    def aux_init__load_pg_config(self,path: pathlib.Path) -> dict:
        with path.open() as f:
            return json.load(f)

    def aux_init__connect_config(self,path:pathlib.Path)-> Engine:
        cfg = self.aux_init__load_pg_config(path)

        url = (
            f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?sslmode={cfg.get('sslmode','require')}"
        )


        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)



    def aux_init__find_config_files(self) -> str:

        candidates = []
        if os.getenv("CONFIG_PATH"):
            candidates.append(pathlib.Path(os.getenv("CONFIG_PATH")))
        candidates.append(pathlib.Path("/app/config/postgres.json"))   # in-container path
        candidates.append(DEFAULT_CFG)

        for p in candidates:
            if p.is_file():
                return p
        raise FileNotFoundError("No postgres.json found. Set CONFIG_PATH or mount /app/config/postgres.json")

        

    def __init__(self):
        path_configuration_file=self.aux_init__find_config_files()
        self.engine=self.aux_init__connect_config(path=path_configuration_file)



def init_schema(engine: Engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id      TEXT PRIMARY KEY,
        signup_date  TIMESTAMPTZ,
        country      TEXT,
        device       TEXT
    );

    CREATE TABLE IF NOT EXISTS events (
        event_id    TEXT PRIMARY KEY,
        user_id     TEXT REFERENCES users(user_id),
        ts          TIMESTAMPTZ,
        event_type  TEXT,
        session_id  TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts DESC);
    CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def append_users(engine: Engine, df: pd.DataFrame) -> None:
    """
    Simple append (good for small/medium batches).
    """
    df.to_sql("users", engine, if_exists="append", index=False)

def append_events(engine: Engine, df: pd.DataFrame) -> None:
    """
    Simple append (good for small/medium batches).
    """
    df.to_sql("events", engine, if_exists="append", index=False)


def _copy_df(engine: Engine, df: pd.DataFrame, table: str) -> None:
    """
    Use PostgreSQL COPY for faster bulk loads.
    Requires psycopg2 driver (installed via SQLAlchemy extras).
    """
    buf = io.StringIO()
    # Write CSV without header
    df.to_csv(buf, index=False, header=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    cols = ",".join(df.columns)
    copy_sql = f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)"
    with engine.begin() as conn:
        raw = conn.connection  # psycopg2 connection
        with raw.cursor() as cur:
            cur.copy_expert(copy_sql, buf)

def copy_users(engine: Engine, df: pd.DataFrame) -> None:
    _copy_df(engine, df, "users")

def copy_events(engine: Engine, df: pd.DataFrame) -> None:
    _copy_df(engine, df, "events")

def event_mix(engine: Engine) -> pd.DataFrame:
    sql = """
    SELECT event_type, COUNT(*) AS n
    FROM events
    GROUP BY event_type
    ORDER BY n DESC;
    """
    return pd.read_sql(sql, engine)

def upsert_users(engine: Engine, df: pd.DataFrame) -> None:
    """
    Upsert into users using a temp table + INSERT ... ON CONFLICT.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE TEMP TABLE tmp_users (LIKE users INCLUDING ALL) ON COMMIT DROP;"))
        # load into temp table quickly (COPY)
        _copy_df(engine, df, "tmp_users")
        conn.execute(text("""
            INSERT INTO users AS u (user_id, signup_date, country, device)
            SELECT user_id, signup_date, country, device FROM tmp_users
            ON CONFLICT (user_id) DO UPDATE
            SET signup_date = EXCLUDED.signup_date,
                country     = EXCLUDED.country,
                device      = EXCLUDED.device;
        """))

def ensure_user_state_columns(engine: Engine):
    with con.connect() as cur:
        cur.exec_driver_sql("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                WHERE table_name='users' AND column_name='is_active') THEN
                ALTER TABLE users ADD COLUMN is_active boolean DEFAULT true;
                UPDATE users SET is_active = true WHERE is_active IS NULL;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                WHERE table_name='users' AND column_name='last_active_ts') THEN
                ALTER TABLE users ADD COLUMN last_active_ts timestamptz NULL;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                WHERE table_name='users' AND column_name='churned_at') THEN
                ALTER TABLE users ADD COLUMN churned_at timestamptz NULL;
            END IF;
        END$$;
        """)
        cur.commit()