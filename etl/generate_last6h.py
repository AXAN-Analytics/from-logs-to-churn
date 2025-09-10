import os, uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

from etl.stores.Postgresql_store import DataBase, init_schema, append_users, append_events
import matplotlib.pyplot as plt


UTC = timezone.utc
rng = np.random.default_rng(42)

def ensure_user_state_columns(con):
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

def diurnal_weights_24h():
    base = np.array([
        0.02,0.015,0.01,0.008,0.008,0.01,  # 0-5
        0.03,0.05,0.06,0.06,0.06,0.06,     # 6-11
        0.05,0.05,0.05,0.05,               # 12-15
        0.06,0.07,0.07,0.06,               # 16-19
        0.045,0.03,0.02,0.015              # 20-23
    ], dtype=float)
    return base / base.sum()

def choose_hours_in_window(now, hours, total):
    w24 = diurnal_weights_24h()
    hours_list = [((now - timedelta(hours=i)).hour) for i in range(hours)][::-1]  # oldest→newest
    weights = np.array([w24[h] for h in hours_list], dtype=float)
    weights = weights / weights.sum()
    counts = rng.multinomial(total, weights)
    return list(zip(range(hours), counts))  # (bucket_index, count)

def sample_timestamps_in_bucket(window_start, bucket_index, count):
    start = window_start + timedelta(hours=bucket_index)
    secs = rng.integers(low=0, high=3600, size=count)
    return [start + timedelta(seconds=int(s)) for s in secs]

import numpy as np

def f(age_index, inactivity_to_life_ratio):
    """
    age_index: 0..1 (0 at signup, 1 at 180+ days)
    inactivity_to_life_ratio: inactive_days / age_days, typically 0..1
    returns: daily hazard (probability) in [0, 0.05]
    """
    ai = np.clip(age_index, 0.0, 1.0)
    r  = np.clip(inactivity_to_life_ratio, 0.0, 1.0)

    # Smooth threshold so inactivity starts to matter after ~20% of lifetime
    t = 0.20       # threshold
    gamma = 1.5    # curvature ( >1 emphasizes higher ratios)
    g = np.clip((r - t) / (1.0 - t), 0.0, 1.0) ** gamma  # 0..1

    base = 0.002                  # 0.20%/day
    age_term = 0.004 * ai         # up to +0.40%/day
    inactivity_max = 0.008        # up to +0.80%/day
    interaction = 0.5 + 0.5 * ai  # inactivity hits harder as age increases


    h = base + age_term + inactivity_max * g * interaction
    return np.clip(h, 0.0, 0.05)  # cap at 5%/day to avoid crazy values


def daily_hazard(age_days, inactive_days):
    # Avoid division by zero for brand-new users
    age_days = np.maximum(age_days, 1e-6)

    age_index = np.minimum(1.0, age_days / 180.0)
    inactivity_to_life_ratio = np.clip(inactive_days / age_days, 0.0, 1.0)

    prob= f(age_index, inactivity_to_life_ratio)

    return age_days,inactive_days,prob


def to_6h_prob(h_daily):
    return 1.0 - (1.0 - h_daily) ** (6.0/24.0)


def get_active_users(con):
    q = """
    SELECT user_id, signup_date, COALESCE(last_active_ts, signup_date) AS last_active_ts
    FROM users
    WHERE is_active = true
    """
    df = pd.read_sql(q, con)
    df["signup_date"] = pd.to_datetime(df["signup_date"], utc=True, errors="coerce")
    df["last_active_ts"] = pd.to_datetime(df["last_active_ts"], utc=True, errors="coerce")

    return df

def insert_new_users(con, n_new, now):
    if n_new <= 0:
        return pd.DataFrame(columns=["user_id","signup_date","country","device","is_active","last_active_ts"])
    countries = ["US","GB","DE","FR","ES","IT","NL","PL","CA","AU"]
    devices   = ["desktop","mobile","tablet"]
    df = pd.DataFrame({
        "user_id": [str(uuid.uuid4()) for _ in range(n_new)],
        "signup_date": [now] * n_new,
        "country": rng.choice(countries, size=n_new, p=[.25,.12,.08,.08,.07,.06,.07,.07,.1,.1]),
        "device": rng.choice(devices,   size=n_new, p=[.55,.40,.05]),
        "is_active": [True] * n_new,
        "last_active_ts": [now] * n_new
    })

    append_users(con, df)
    return df


def insert_dummy_users(con, n_new, now):
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



def plot_churn_map(data, title="Churn probability by age & inactivity"):
    """
    data: list of (age_days, inactive_days, pb_churn)
          pb_churn should be a probability (0..1). If some values look like 0..100, we'll normalize.
    """
    if len(data) == 0:
        raise ValueError("Empty data list")

    arr = np.array(data, dtype=float)
    age = arr[:, 0]
    inactive = arr[:, 1]
    p = arr[:, 2]

    # Normalize if values look like percentages (e.g., >1)
    if np.nanmax(p) > 1.0:
        p = p / 100.0

    # Basic sanity clamps
    age = np.clip(age, 0, None)
    inactive = np.clip(inactive, 0, None)
    p = np.clip(p, 0.0, 1.0)

    plt.figure()
    sc = plt.scatter(age, inactive, c=p, s=20, alpha=0.9)  # uses default colormap
    cbar = plt.colorbar(sc)
    cbar.set_label("Churn probability")

    plt.xlabel("Age (days)")
    plt.ylabel("Inactive days")
    plt.title(title)

    # Optional: show y=x reference (inactive cannot exceed age in theory)
    # xmax = np.nanmax(age) if np.isfinite(np.nanmax(age)) else 1
    # plt.plot([0, xmax], [0, xmax])

    plt.tight_layout()
    plt.show()



def churn_some_users(con, active_df, now):
    if active_df.empty:
        return 0
    age_days = ((now - active_df["signup_date"]).dt.total_seconds() / 86400.0).clip(lower=0)
    inactive_days = ((now - active_df["last_active_ts"]).dt.total_seconds() / 86400.0).clip(lower=0)

    val=[daily_hazard(int(a), int(i)) for a, i in zip(age_days, inactive_days)]
    #plot_churn_map(val)
  
    p6 = [to_6h_prob(i[2]) for i in val]
    flags = rng.random(len(active_df)) < np.array(p6)
    to_churn = active_df.loc[flags, "user_id"].tolist()
    if not to_churn:
        return 0
    with con.connect() as cur:
        cur.exec_driver_sql("""
            UPDATE users u
            SET is_active = false, churned_at = %s
            FROM (SELECT UNNEST(%s::uuid[]) AS uid) x
            WHERE u.user_id::uuid = x.uid
        """, (now, to_churn))
        cur.commit()
    return len(to_churn)

def generate_events_last_6h(con, X_daily, now):
    H = 6
    window_start = now - timedelta(hours=H)
    active = get_active_users(con)
    n_active = len(active)

    target = max(0, int(round(X_daily * H / 24.0)))
    if n_active == 0 or target == 0:
        return pd.DataFrame(columns=["event_id","user_id","ts","event_type","session_id"])

    # recency weights (every week, weight is halved)
    recency_days = ((now - active["last_active_ts"]).dt.total_seconds() / 86400.0).clip(lower=0)
    weights = np.power(0.5, recency_days / 7.0)
    weights = weights / weights.sum()
    user_ids = active["user_id"].to_numpy()

    rows = []
    buckets = choose_hours_in_window(now, H, target)
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
        with con.connect() as cur:
            cur.exec_driver_sql("""
                UPDATE users
                SET last_active_ts = %s
                WHERE user_id = ANY(%s)
            """, (now, list(seen)))
            cur.commit()
    return events

def run_last_6h(X_daily: int, Z_per_6h: int, now: Optional[datetime] = None) -> dict:
    if now is None:
        now = datetime.now(timezone.utc)
    db = DataBase()

    con= db.engine

    init_schema(con)              # must create users/events if missing (see DDL above)
    ensure_user_state_columns(con)

    # arrivals
    z = int(np.random.poisson(max(0, Z_per_6h)))

    new_users = insert_new_users(con, z, now)
    # churn
    active_before = get_active_users(con)


    t = churn_some_users(con, active_before, now)


    # events in last 6h
    events = generate_events_last_6h(con, X_daily, now)

    
    if not events.empty:
        append_events(con, events)

    # return a small summary
    active_after = pd.read_sql("SELECT COUNT(*) FROM users WHERE is_active=true", con).iloc[0,0]
    return {"now": now.isoformat(), "new_users": len(new_users), "churned": t,
            "events_6h": len(events), "active_after": int(active_after)}


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
