
from flask import Flask, jsonify
import sqlite3, pandas as pd

app = Flask(__name__)
DB = "data/warehouse/saas.db"

def df_to_json(df):
    return [dict(zip(df.columns, row)) for row in df.itertuples(index=False, name=None)]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    con = sqlite3.connect(DB)
    dau = pd.read_sql("select * from daily_active_users order by date", con, parse_dates=["date"])
    users = pd.read_sql("select count(*) as users from users", con)
    churn = pd.read_sql("select avg(is_churned_30d)*100 as churn_rate from users_enriched", con)
    con.close()
    return {
        "total_users": int(users.iloc[0,0]),
        "avg_dau": int(dau["dau"].mean()),
        "churn_rate_30d_pct": round(float(churn.iloc[0,0]), 2),
        "dau": df_to_json(dau)
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
