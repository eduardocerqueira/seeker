#date: 2022-07-08T17:05:15Z
#url: https://api.github.com/gists/76331fa1eecee0780d7c0ba4707d98e6
#owner: https://api.github.com/users/zoeding-lyft

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import json
import pandas as pd

cdc_df = pd.read_csv('cdc_prod.csv')
tq_dropped_off_df = pd.read_csv('tq_dropped_off_prod.csv')
tq_completed_df = pd.read_csv('tq_completed_prod.csv')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("cdc records count:", cdc_df.shape[0])
    print("\ntq_dropped_off records count:", tq_dropped_off_df.shape[0])
    print("\ntq_completed records count:", tq_completed_df.shape[0])

    cdc_ride_id_list = list(set(cdc_df["driverridesinfo.ride_id"]))
    tq_dropped_off_list = [int(x) for x in tq_dropped_off_df["driverridesinfo.ride_id"]]
    tq_completed_list = [int(x) for x in tq_completed_df["driverridesinfo.ride_id"]]
    tq_ride_id_set = set(tq_completed_list + tq_dropped_off_list)

    print("\nTotal number of ride_id from TQ: ", len(tq_ride_id_set))
    print("\nTotal number of ride_id from CDC: ", len(cdc_ride_id_list))

    not_in_tq_set = set()

    for ride_id in cdc_ride_id_list:

        if ride_id not in tq_ride_id_set:
            not_in_tq_set.add(ride_id)

    print("\nThe id that in cdc but not in tq: ", len(not_in_tq_set))
    print(not_in_tq_set)

