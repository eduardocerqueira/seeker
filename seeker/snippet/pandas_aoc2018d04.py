#date: 2023-06-29T16:48:23Z
#url: https://api.github.com/gists/2efafb3771fc47854731bff60a7e54e9
#owner: https://api.github.com/users/RussellWaite

import pandas as pd

schedule = pd.read_fwf("input", colspecs=[(1,11),(12,14),(15,17),(19, 25), (25,31) ], header=None, names=["date", "hour", "minute", "action", "id"]) #names=["reading"],
schedule['action'] = schedule['action'].str.lower()
schedule = schedule.sort_values(by=['date', 'hour', 'minute'])

# fill in ID for every row
schedule['id'] = schedule['id'].str.replace('[^0-9]', '',regex=True)
schedule['id'] = schedule.apply(lambda x: x['id'] if x['id'] != '' else None, axis=1)
schedule['id'] = schedule['id'].ffill(axis=0)
schedule['time_diff'] = schedule['minute'].diff()

# just get falls and wakes
sleeps = schedule.query("action == 'falls' or action == 'wakes'")
wake_times = sleeps.query("action == 'wakes'")

# temp list to populate then build a dataframe from
tmp_guard_asleep = [[]]
def generate_guard_asleep(row):
    for x in range((row['minute']-int(row['time_diff'])), row['minute'], 1):
        tmp_guard_asleep.append([row['id'], x])

# expand time range into guard id and minute asleep.
wake_times.apply(generate_guard_asleep, axis=1)

guard_asleep = pd.DataFrame( tmp_guard_asleep, columns=["id","minute"])
heaviest_sleeper = guard_asleep.groupby(['id']).count()['minute'].idxmax()
minute_slept_aggregate = guard_asleep.query('id == @heaviest_sleeper')
most_slept_minute = int(minute_slept_aggregate.groupby(['minute']).count()['id'].idxmax())

print('part 1: ', int(heaviest_sleeper) * most_slept_minute)
# should be 99911

[guard_id, minute] = guard_asleep.groupby(['id', 'minute'])['minute'].count().idxmax()
print("part 2: ", int(guard_id) * int(minute))
# should be 65854