# scripts/reset_db.py
from etl.stores.Postgresql_store import connect_from_config, init_schema

def reset():
    con = connect_from_config()
    with con.cursor() as cur:
        # Drop child first, then parent
        cur.execute("DROP TABLE IF EXISTS events;")
        cur.execute("DROP TABLE IF EXISTS users;")
    con.commit()
    # Recreate from your init_schema (idempotent)
    init_schema(con)
    con.commit()
    con.close()
    print("DB reset: users/events dropped and recreated.")

if __name__ == "__main__":
    reset()
