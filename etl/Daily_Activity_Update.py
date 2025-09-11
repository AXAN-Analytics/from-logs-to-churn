import os, uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

from etl.stores.Postgresql_store import DataBase, init_schema, ensure_user_state_columns, append_users, append_events
import matplotlib.pyplot as plt


from Daily_Activity_Update_auxillary import daily_hazard,to_6h_prob
from Daily_Activity_Update_auxillary import choose_hours_in_window,sample_timestamps_in_bucket

UTC = timezone.utc
rng = np.random.default_rng(42)


### I call the class daily but it runs 4 times a day


class Daily_Activity_Update:
    def __init__(self,now: Optional[datetime] = None):
        
        self.now = datetime.now(timezone.utc)  if now is None else now
        self.daily_frequency=4

        self.expected_event_daily=25_000
        self.expected_new_users_daily=16
        
        self.database=DataBase()
        self.engine=self.database.engine

        init_schema(self.engine)             
        ensure_user_state_columns(self.engine)




    def init_real_values(self):

        self.real_event_daily=int(np.random.poisson(max(0, self.expected_event_daily)))
        self.real_new_user_daily=int(np.random.poisson(max(0, self.expected_new_users_daily)))


    def __call__(self):
        new_users = self.insert_new_users()
        active_before = self.get_active_users()
        nb_user_churned = self.churn_some_users(active_before)
        events = self.generate_events_last_6h()

        
        if not events.empty:
            append_events(self.engine, events)

        # return a small summary
        active_after = pd.read_sql("SELECT COUNT(*) FROM users WHERE is_active=true", self.engine).iloc[0,0]
        return {
            "now": self.now.isoformat(), 
            "new_users": len(new_users), 
            "churned": t,
            "events_6h": len(events), 
            "active_after": int(active_after)
        }



    def insert_new_users(self):

        if self.real_new_user_daily <= 0:
            return pd.DataFrame(columns=["user_id","signup_date","country","device","is_active","last_active_ts"])
        countries = ["US","GB","DE","FR","ES","IT","NL","PL","CA","AU"]
        devices   = ["desktop","mobile","tablet"]
        df = pd.DataFrame({
            "user_id": [str(uuid.uuid4()) for _ in range(self.real_new_user_daily)],
            "signup_date": [self.now] * self.real_new_user_daily,
            "country": rng.choice(countries, size=self.real_new_user_daily, p=[.25,.12,.08,.08,.07,.06,.07,.07,.1,.1]),
            "device": rng.choice(devices,   size=self.real_new_user_daily, p=[.55,.40,.05]),
            "is_active": [True] * self.real_new_user_daily,
            "last_active_ts": [self.now] * self.real_new_user_daily
        })

        append_users(self.engine, df)
        return df


    
    def get_active_users(self):
        q = """
            SELECT user_id, signup_date, COALESCE(last_active_ts, signup_date) AS last_active_ts
            FROM users
            WHERE is_active = true
        """
        df = pd.read_sql(q, self.engine)
        df["signup_date"] = pd.to_datetime(df["signup_date"], utc=True, errors="coerce")
        df["last_active_ts"] = pd.to_datetime(df["last_active_ts"], utc=True, errors="coerce")

        return df
    
    
    def churn_some_users(self,active_df:pd.DataFrame):
        if active_df.empty:
            return 0
        

        age_days = ((self.now - active_df["signup_date"]).dt.total_seconds() / 86400.0).clip(lower=0)
        inactive_days = ((self.now - active_df["last_active_ts"]).dt.total_seconds() / 86400.0).clip(lower=0)

        p6 = [to_6h_prob(daily_hazard(int(a), int(i)),self.daily_frequency) for a, i in zip(age_days, inactive_days)]
        flags = rng.random(len(active_df)) < np.array(p6)
        to_churn = active_df.loc[flags, "user_id"].tolist()
        if not to_churn:
            return 0
        

        with self.engine.connect() as cur:
            cur.exec_driver_sql("""
                UPDATE users u
                SET is_active = false, churned_at = %s
                FROM (SELECT UNNEST(%s::uuid[]) AS uid) x
                WHERE u.user_id::uuid = x.uid
            """, (self.now, to_churn))
            cur.commit()
        return len(to_churn)




    def generate_events_last_6h(self):
       
        window_start = self.now - timedelta(hours=H)
        active = self.get_active_users(self.engine)
        n_active = len(active)

        target = max(0, int(round(self.real_event_daily * self.daily_frequency / 24.0)))
        if n_active == 0 or target == 0:
            return pd.DataFrame(columns=["event_id","user_id","ts","event_type","session_id"])

        # recency weights (every week, weight is halved)
        recency_days = ((self.now - active["last_active_ts"]).dt.total_seconds() / 86400.0).clip(lower=0)
        weights = np.power(0.5, recency_days / 7.0)
        weights = weights / weights.sum()
        user_ids = active["user_id"].to_numpy()

        rows = []
        buckets = choose_hours_in_window(self.now, int(24/self.daily_frequency), target)
        for b_idx, count in buckets:
            if count == 0: 
                continue
            ts_list = sample_timestamps_in_bucket(window_start, b_idx, count)
            # derive sessions (avg ~5 events/session)   <<-------------------------------------------------- PARAMETER
            sessions = max(1, count // 5)
            owners = rng.choice(user_ids, size=sessions, replace=True, p=weights)
            # split counts across sessions
            parts = rng.multinomial(count, np.ones(sessions)/sessions)
            i = 0
            for owner, n in zip(owners, parts):
                if n == 0: 
                    continue
                session_id = str(uuid.uuid4())
                for _ in range(n):
                    r = rng.random()
                    et = "page_view"
                    if r < 0.15: et = "login" # <------------------------ P(login)=0.15
                    elif r < 0.18: et = "purchase" # <--------------------------- P(purchase)
                    elif r < 0.30: et = "email_open"
                    rows.append([str(uuid.uuid4()), owner, ts_list[i], et, session_id])
                    i += 1

        events = pd.DataFrame(rows, columns=["event_id","user_id","ts","event_type","session_id"])



        # update last_active_ts for users that appeared
        if not events.empty:
            seen = tuple(events["user_id"].unique())
            with self.engine.connect() as cur:
                cur.exec_driver_sql("""
                    UPDATE users
                    SET last_active_ts = %s
                    WHERE user_id = ANY(%s)
                """, (self.now, list(seen)))
                cur.commit()
        return events






if __name__=='__main__':

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    # knobs
    X_DAILY = 25000    # your value
    Z_PER_6H = 16    # your value

    tz = ZoneInfo("Europe/Amsterdam")
    now_ref = datetime.now(tz).replace(minute=0, second=0, microsecond=0)

    start = now_ref - timedelta(days=180)
    t = (start + timedelta(hours=6)).replace(minute=0, second=0, microsecond=0)


    it=0
    while t <= now_ref:
        it+=1
        run_last_6h(X_DAILY, Z_PER_6H, now=t)
        print(f'Data at {t} generated')
        t += timedelta(hours=6)

        if it>180*4:
            exit()
