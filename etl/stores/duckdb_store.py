from __future__ import annotations
import os, pathlib, duckdb, pandas as pd

DEFAULT_PATH = "data/warehouse.duckdb"

def connect(db_path: str = DEFAULT_PATH) -> duckdb.DuckDBPyCovenvnnection:
    p=pathlib.Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(p))


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
       CREATE TABLE IF NOT EXISTS users (
                user_id TEXT,
                signup_date TIMESTAMP,
                country TEXT,
                device TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
                event_id TEXT,
                user_id TEXT,
                ts TIMESTAMP,
                event_type TEXT,
                session_id TEXT           
        );        
                
                
    """)


def append_users(con: duckdb.DuckDBPyConnection, df:pd.DataFrame)-> None:
    con.register("u",df)
    con.execute("INSERT INTO users SELECT * FROM u")


def append_events(con: duckdb.DuckDBPyConnection, df:pd.DataFrame)-> None:
    con.register("e",df)
    con.execute("INSERT INTO events SELECT * FROM e")


def event_mix(con: duckdb.DuckDBPyConnection, df:pd.DataFrame)-> None:
    return con.execute("""
        SELECT event_type, COUNT(*) AS n
        FROM events
        GROUP BY 1 ORDER BY 2 DESC                
    """).df()