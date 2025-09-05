import os,json, pathlib
import pandas as pd
from sqlalchemy import text
from etl.stores.Postgresql_store import connect_from_config, init_schema, append_users, append_events, event_mix

# Prefer XDG path; allow override via env CONFIG_PATH
DEFAULT_CFG = pathlib.Path.home() / ".config" / "OVH-cloud" / "from-logs-to-churn" / "postgres.json"


def load_cfg():
    candidates = []
    if os.getenv("CONFIG_PATH"):
        candidates.append(pathlib.Path(os.getenv("CONFIG_PATH")))
    candidates.append(pathlib.Path("/app/config/postgres.json"))   # in-container path
    candidates.append(DEFAULT_CFG)

    for p in candidates:
        if p.is_file():
            return connect_from_config(path=p)
    raise FileNotFoundError("No postgres.json found. Set CONFIG_PATH or mount /app/config/postgres.json")

def main():
    engine = load_cfg()
    print(event_mix(engine))

if __name__ == "__main__":
    main()
