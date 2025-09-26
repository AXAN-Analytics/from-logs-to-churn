import os, uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

from etl.stores.Postgresql_store import DataBase, init_schema, append_users, append_events
import matplotlib.pyplot as plt


UTC = timezone.utc
rng = np.random.default_rng(42)


from etl.Daily_Activity_Update import Daily_Activity_Update

class Daily_Activity_Update_Simulation(Daily_Activity_Update):


    def insert_new_users(con, n_new, now):
        if n_new <= 0:
            return pd.DataFrame(columns=["user_id","signup_date","country","device","is_active","last_active_ts"])

        countries = ["US","GB","DE","FR","ES","IT","NL","PL","CA","AU"]
        devices   = ["desktop","mobile","tablet"]
        df = pd.DataFrame({
            "user_id": [str(uuid.uuid4()) for _ in range(n_new)],
            "country": rng.choice(countries, size=n_new, p=[.25,.12,.08,.08,.07,.06,.07,.07,.1,.1]),
            "device": rng.choice(devices,   size=n_new, p=[.55,.40,.05]),
            "is_active": [True] * n_new
        })


        start_6m = now - pd.DateOffset(months=6)
        start_3m = now - pd.DateOffset(months=3)

        # Make randoms as Series (key change)
        u = pd.Series(rng.random(n_new), index=df.index)
        v = pd.Series(rng.random(n_new), index=df.index)

        # 1) signup_date ~ Uniform[ now-6m , now )
        signup_date = start_6m + (now - start_6m) * u

        # 2) Lower bound for last_active_ts is max(signup_date, now-3m)
        lb = signup_date.where(signup_date > start_3m, start_3m)

        # 3) last_active_ts ~ Uniform[ lb , now ]
        last_active_ts = lb + (now - lb) * v

        # 4) Ensure strict signup_date < last_active_ts and clamp to now
        eps = pd.to_timedelta(1, "s")
        last_active_ts = last_active_ts.where(last_active_ts > signup_date, signup_date + eps)
        last_active_ts = last_active_ts.where(last_active_ts <= now, now)

        # 5) Assign back
        df["signup_date"] = signup_date
        df["last_active_ts"] = last_active_ts

        # (optional) checks
        assert (df["signup_date"] < df["last_active_ts"]).all()
        assert (df["last_active_ts"] >= start_3m).all()
        assert (df["signup_date"] >= start_6m).all()


        append_users(con, df)
        return df