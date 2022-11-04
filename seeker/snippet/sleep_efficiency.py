#date: 2022-11-04T17:08:08Z
#url: https://api.github.com/gists/43c1196e7986dbd07fa0b926eafc0c74
#owner: https://api.github.com/users/hayesall

# MIT License

"""
Questions:

- What do sleep metrics look like at or near visit 2?
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


mapping = pd.read_csv("data/mapping-hmc-id-garmin.csv")
EGAs = pd.read_csv("data/egas.csv", parse_dates=[4, 10, 16], na_values=["."])
sleep_users = pd.DataFrame({"garmin_guid": [file.replace(".csv","") for file in os.listdir("data/sleep/")]})


EGAs = EGAs[["Record ID:", "V1\nBiometric\nDate", "V2\nBiometric\nDate", "EGA at\nV1\nBiometric\nDate", "EGA at\nV2\nBiometric\nDate", "Infant DOB"]]
EGAs.columns = ["RecordID", "V1_Date", "V2_Date", "V1_EGA", "V2_EGA", "Infant_DOB"]

EGAs.dropna(inplace=True)


def n_weeks_to_n_days(n_weeks):
    return round(n_weeks * 7)

EGAs["EGA_Days"] = pd.to_timedelta(EGAs["V1_EGA"].apply(n_weeks_to_n_days), unit="days")
EGAs["Conception"] = EGAs["V1_Date"] - EGAs["EGA_Days"]

EGAs["2nd_tri"] = EGAs["Conception"] + pd.to_timedelta(13 * 7, unit="days")
EGAs["3rd_tri"] = EGAs["Conception"] + pd.to_timedelta(26 * 7, unit="days")

map_merge = pd.merge(mapping, sleep_users, on="garmin_guid")

EGAs = pd.merge(map_merge, EGAs, on="RecordID")


def process_sleep(df):
    return df["level"].mean()


results = []

for user in tqdm(EGAs["garmin_guid"]):

    df = pd.read_csv("data/sleep/" + user + ".csv", parse_dates=[0], index_col=[0])

    user_data = EGAs[EGAs["garmin_guid"] == user]

    v2_date = pd.to_datetime(user_data["V2_Date"].values[0])
    v2_2_weeks = v2_date + pd.Timedelta(days=14)

    between_query = (v2_date < df.index) & (df.index < v2_2_weeks)

    selection = df[between_query]

    if len(selection) > 100:
        results.append(len(selection))

print(len(results))
plt.hist(results)
plt.show()
