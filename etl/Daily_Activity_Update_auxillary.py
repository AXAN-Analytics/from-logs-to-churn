import os, uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

from etl.stores.Postgresql_store import DataBase, init_schema, append_users, append_events
import matplotlib.pyplot as plt


UTC = timezone.utc
rng = np.random.default_rng(42)

def diurnal_weights_24h():

    ### this is the probability at each of our of the day to have traffic
    base = np.array([
        0.02,0.015,0.01,0.008,0.008,0.01,  # 0-5
        0.03,0.05,0.06,0.06,0.06,0.06,     # 6-11
        0.05,0.05,0.05,0.05,               # 12-15
        0.06,0.07,0.07,0.06,               # 16-19
        0.045,0.03,0.02,0.015              # 20-23
    ], dtype=float)
    return base / base.sum()

def choose_hours_in_window(now, hours, total):

    #### this is the generated hour of the day for the traffic activity to happen
    w24 = diurnal_weights_24h()
    hours_list = [((now - timedelta(hours=i)).hour) for i in range(hours)][::-1]  # 6 last hours : from 6th to last hour -> 6 has been standarduzed as hours
    weights = np.array([w24[h] for h in hours_list], dtype=float) # list of probability over last six hours
    weights = weights / weights.sum() # normalization of probability
    counts = rng.multinomial(total, weights) # choose traffic hours given the probability for a total number of "total" events
    return list(zip(range(hours), counts))  # aggregating result

def sample_timestamps_in_bucket(window_start, bucket_index, count):

    #### this is a timestamp generator.. 
    #### We give a count of events (multiple event occur for a user, like "login","visit" ,"purchase")
    #### Then we select random timestamp within the hour for this user

    start = window_start + timedelta(hours=bucket_index)
    secs = rng.integers(low=0, high=3600, size=count)
    return [start + timedelta(seconds=int(s)) for s in secs]




def daily_hazard(age_days, inactive_days):
    # Daily hasard is the  risk for a given user with a certain registration age (age_days) and a certain inactivity time (inactivity_days)
    # to be decided according to a random risk algorithm as : will not return


    age_days = np.maximum(age_days, 1e-6)

    ai = np.clip(age_days/180.0, 0.0, 1.0)   ### normalized user age, beyond 6 month of activity the user is categorized as "installed"
    r  = np.clip(inactive_days / age_days, 0.0, 1.0) ### late absence duration over lifespan for a user

    # Smooth threshold so inactivity starts to matter after ~20% of lifetime
    t = 0.20       # threshold
    gamma = 1.5    # curvature ( >1 emphasizes higher ratios)
    g = np.clip((r - t) / (1.0 - t), 0.0, 1.0) ** gamma  # 0..1


    ### t act as a low limit : below t being 20% a user is considered as not having been too "inactive".
    ### we raise the result to the power 1.5 in order to make long inactivity over lifespan for a user sharply increasing the risk of churn  at the end: assuming a brutal increase for users with just a few beginning visits

    base = 0.002                  # every user has a 0.2% churn risk
    age_term = 0.004 * ai         # the longevity of a user increasing the risk of churn up to 0.4% for installed users (linearly)
    inactivity_max = 0.008        # the late absence of a user over their lifespan also increase the risk up to 0.8% for a user (non linearly)
    interaction = 0.5 + 0.5 * ai  # inactivity hits harder as age increases


    h = base + age_term + inactivity_max * g * interaction ### age adjusted multiplied by inactivity over lifespan adjusted give 
    return np.clip(h, 0.0, 0.05)  # cap at 5%/day to avoid crazy values


def to_6h_prob(h_daily,daily_frequency):

    ### scale down probability according to daily_frequency

    return 1.0 - (1.0 - h_daily) ** (daily_frequency/24.0)
