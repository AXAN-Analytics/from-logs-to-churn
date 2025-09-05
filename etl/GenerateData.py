
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import uuid, random, os

from typing import Dict, List
import logging
import argparse
import inspect

#from etl.stores.duckdb_store import connect, init_schema, append_users, append_events
from etl.stores.Postgresql_store import connect_from_config, init_schema, append_users, append_events

### Argument Parsing

ap = argparse.ArgumentParser()
ap.add_argument(
    "--n-users", "--N_USERS", "--N_USRES",
    type=int, dest="n_users",
    help="Number of users to generate (overrides env N_USERS)"
)
args = ap.parse_args()


### Logging initialization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etl.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)



def catch_local_variables(*args):
    frame = inspect.currentframe().f_back 
    return {k: frame.f_locals[k] for k in args if k in frame.f_locals}


class GenerateData:


    def init__time_frame(self):

        END = pd.Timestamp.today().normalize()
        START = END - pd.DateOffset(months=12)

        self.parameters.update(catch_local_variables("START","END"))
        
    def init__user_info(self,n_users):
        
        N_USERS = n_users # int(os.environ.get("N_USERS", "50000"))
        countries = ["US","GB","DE","FR","ES","IT","NL","PL","CA","AU"]
        devices = ["desktop","mobile","tablet"]

        self.parameters.update(catch_local_variables("N_USERS","countries","devices"))



    def log__user_parameter_info(self)->None:
        x=self.parameters
        N_USERS,START,END= x['N_USERS'],x['START'],x['END'] 
        logger.info("Starting data generation: N_USERS=%d, window=%s to %s", N_USERS, START.date(), END.date())

    def log__user_generared_info(self,users:pd.DataFrame)->None:

        
        ### users.colnames= ("user_id","signup_date","country","device")
        logger.info("Generated %d users across %d countries", len(users), users['country'].nunique())
        logger.info("Generating events for %d users...", len(users))

    def log__event_generated_info(self,events:pd.DataFrame,users:pd.DataFrame)->None:
        
        
        ### users.colnames= ("user_id","signup_date","country","device")
        ### events.colnames= ("event_id","user_id","ts","event_type","session_id
        
        logger.info("Generated %d events total", len(events))
        logger.info("Saving users.csv (%d rows) and events.csv (%d rows) into data/raw/", len(users), len(events))

    def log__saved_data(self,events:pd.DataFrame,users:pd.DataFrame)->None:

        logger.info("Generated:", users.shape, events.shape)
        logger.info("Files: data/raw/users.csv, data/raw/events.csv")



    def __init__(self,n_users)->None:

        os.makedirs("data/raw", exist_ok=True)
        np.random.seed(42)
        random.seed(42)
        self.fake = Faker()


        self.parameters = dict()
        self.init__time_frame()
        self.init__user_info(n_users)
        

    def make_users(self, n: int | None = None) -> pd.DataFrame:
        START: pd.Timestamp = self.parameters["START"]  # type: ignore[assignment]
        END: pd.Timestamp = self.parameters["END"]      # type: ignore[assignment]
        countries: List[str] = self.parameters["countries"]  # type: ignore[assignment]
        devices: List[str] = self.parameters["devices"]      # type: ignore[assignment]
        N_USERS: int = self.parameters["N_USERS"]            # type: ignore[assignment]

        n = n or N_USERS

        signup_dates = pd.to_datetime(
            np.random.choice(pd.date_range(START, END - pd.DateOffset(days=60)), size=n)
        )
        df = pd.DataFrame({
            "user_id": [str(uuid.uuid4()) for _ in range(n)],
            "signup_date": signup_dates,
            "country": np.random.choice(countries, size=n, p=[.25,.12,.08,.08,.07,.06,.07,.07,.1,.1]),
            "device": np.random.choice(devices, size=n, p=[.55,.4,.05]),
        })

        return df


    def log__event_generation(self,id,rows):
        if id!=0 and id%1000==0:
            logger.info(f"{id} users analysed :{round(len(rows)/id,2)} events per user in average")
 


    def make_events(self, users: pd.DataFrame) -> pd.DataFrame:

        ### users.colnames= ("user_id","signup_date","country","device")

        END: pd.Timestamp = self.parameters["END"]  # type: ignore[assignment]

        rows = []
        for id, u in users.iterrows():

            self.log__event_generation(id,rows)

            months_active = max(1, (END.to_pydatetime().date() - u.signup_date.date()).days // 30)
            base = np.clip(np.random.normal(6, 3), 1, 20)  # avg sessions/month

            for m in range(months_active):
                month_start = (u.signup_date + pd.DateOffset(months=m)).normalize()
                if month_start > END: break

                # activity probability decays over cohorts
                if np.random.rand() < max(0.15, 0.95 - 0.04*m):
                    sessions = np.random.poisson(base)
                    for _ in range(sessions):
                        day_offset = int(np.random.randint(0, max(1,(END - month_start).days+1)))
                        ts = month_start + pd.DateOffset(days=day_offset) + pd.Timedelta(minutes=np.random.randint(8*60, 22*60))
                        session_id = str(uuid.uuid4())

                        # multiple page views per session
                        for _ in range(np.random.randint(3, 8)):
                            rows.append([str(uuid.uuid4()), u.user_id, ts + pd.Timedelta(minutes=np.random.randint(0,30)), "page_view", session_id])

                        # some logins/purchases/email opens
                        if np.random.rand() < 0.6:
                            rows.append([str(uuid.uuid4()), u.user_id, ts, "login", session_id])
                        if np.random.rand() < 0.05:
                            rows.append([str(uuid.uuid4()), u.user_id, ts + pd.Timedelta(minutes=5), "purchase", session_id])
                        if np.random.rand() < 0.25:
                            rows.append([str(uuid.uuid4()), u.user_id, ts + pd.Timedelta(minutes=2), "email_open", session_id])

        cols = ["event_id","user_id","ts","event_type","session_id"]
        return pd.DataFrame(rows, columns=cols)

    def pull_data_to_DB(self,users: pd.DataFrame, events:pd.DataFrame) -> None:

        con = connect_from_config()
        init_schema(con)
        append_users(con,users)
        append_events(con,events)

        logger.info("Data sent to DB on OVH Cloud (location Gravelines)")



    def save_data(self, users: pd.DataFrame, events: pd.DataFrame) -> None:

        
        ### users.colnames= ("user_id","signup_date","country","device")
        ### events.colnames= ("event_id","user_id","ts","event_type","session_id")
  
        users.to_csv("data/raw/users.csv", index=False)
        events.to_csv("data/raw/events.csv", index=False)

        logger.info("Data stored locally")

if __name__ == "__main__":

    data_generator=GenerateData(n_users=args.n_users)
    data_generator.log__user_parameter_info()

    users=data_generator.make_users()
    data_generator.log__user_generared_info(users)

    events=data_generator.make_events(users)
    data_generator.log__event_generated_info(users,events)

    data_generator.pull_data_to_DB(users,events)
    #data_generator.save_data(users,events)
    #data_generator.log__saved_data(users,events)

    