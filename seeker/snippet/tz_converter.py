#date: 2023-11-30T16:56:34Z
#url: https://api.github.com/gists/a2ab8e7160ce9d540885b1fded08d13d
#owner: https://api.github.com/users/kongmunist

# This file uses the sleep stress file's timestamp to convert each sleep row to its proper timezone
import pandas as pd
import matplotlib.pyplot as plt
import copy

stressFile = "cooked/Stress_Stress Score.csv"
sleepFile = "cooked/Sleep_sleep.csv"

# Each line in sleep has a few time-associated cols: (dateOfSleep, startTime, endTime)
# These are given in the timezone of the user's device and not in UTC
# However, stress is reported in UTC, ~0-30 minutes after the sleep ends. We can use this timestamp and the endTime timestamp to get the timezone offset, and to correct the sleep times for UTC

# Load the stress file
stress = pd.read_csv(stressFile)
sleep = pd.read_csv(sleepFile)

# Check how many duplicates in dateOfSleep
print("Duplicates in dateOfSleep: ", sleep.duplicated(subset=["dateOfSleep"]).sum())

# stress "DATE" lines up with sleep "dateOfSleep"
# Get sleep rows that happened at the same date as a stress row
allStressSleepPairs = []
for i, stressRow in stress.iterrows():
    stressDate = stressRow['DATE']
    allStressSleepPairs.append([stressRow['UPDATED_AT'], sleep[sleep['dateOfSleep'] == stressDate]])
# sanity check totals
print("Total stress rows: ", len(stress))
print("Total sleep rows matched to a stress date: ", sum([x[1].shape[0] for x in allStressSleepPairs]))

# For each stress row, get the sleep row differences
offsetsWithMin = []
offsets = []
addThisToSleepCSV = []
for stressEl, sleepEl in allStressSleepPairs:
    s1 = pd.to_datetime(stressEl,format="mixed")
    s2 = pd.to_datetime(sleepEl['endTime'])

    differ = s1-s2
    minInd = abs(differ).argmin()

    offsetsWithMin.append([s1, s2, differ.min()])
    offsets.append([differ.index[minInd], s2, differ, differ.iloc[minInd], differ.iloc[minInd].total_seconds()/3600])

    tmp = differ.iloc[minInd].total_seconds()/3600
    tmp = int(tmp - (tmp < 0) )#+ (tmp > 0))

    addThisToSleepCSV.append([differ.index[minInd], tmp, s1, s2])
for x in offsets:
    print("\n",x)
for x in addThisToSleepCSV:
    print("\n",x)

for x in offsetsWithMin:
    # print each in another color
    print(f"\[\033[1;32m{x[0]}\033[0m\] \[\033[1;31m{x[1]}\033[0m\] \[\033[1;34m{x[2]}\033[0m\]")



# Show the offsets graphically so we can find a way to remove outliers
# timestamps = [x[2] for x in addThisToSleepCSV]

### No filtering
tzOffs = [x[1] for x in addThisToSleepCSV]
figure, ax = plt.subplots(figsize=(12,5))
ax.plot(tzOffs, marker=".")
ax.hlines(0, 0, len(tzOffs), colors='r', linestyles='dashed')
plt.title("Timezone offsets for sleep rows, unfiltered")



# ### Easy filtering
# nowsleepcsv = []
# for row in addThisToSleepCSV:
#     if row[1] > -12 and row[1] < 12:
#         nowsleepcsv.append(copy.deepcopy(row))
# print("new len vs old len: ", len(nowsleepcsv), len(addThisToSleepCSV))

# tzOffs = [x[1] for x in nowsleepcsv]
# figure, ax = plt.subplots(figsize=(12,5))
# ax.plot(tzOffs, marker=".")
# ax.hlines(0, 0, len(tzOffs), colors='r', linestyles='dashed')
# plt.title("Timezone offsets for sleep rows, removing |offset| > 12")



nowsleepcsv = copy.deepcopy(addThisToSleepCSV)


tzOffs = [x[1] for x in nowsleepcsv]
figure, ax = plt.subplots(figsize=(12,5))
ax.plot(tzOffs, marker=".")
ax.hlines(0, 0, len(tzOffs), colors='r', linestyles='dashed')
plt.title("Timezone offsets for sleep rows, removing |offset| > 12")

### if a point is <2hr different from the before and after, change it
for i in range(1, len(nowsleepcsv)-1):
    if (nowsleepcsv[i-1][1] == nowsleepcsv[i+1][1]): # if before and after equal
        if (abs(nowsleepcsv[i][1] - nowsleepcsv[i-1][1]) != 0): # and this one is different
        # if (abs(nowsleepcsv[i][1] - nowsleepcsv[i-1][1]) < 2):
            nowsleepcsv[i][1] = nowsleepcsv[i-1][1] # Replace it

tzOffs = [x[1] for x in nowsleepcsv]
figure, ax = plt.subplots(figsize=(12,5))
ax.plot(tzOffs, marker=".")
ax.set_ylim([-12,12])
ax.hlines(0, 0, len(tzOffs), colors='r', linestyles='dashed')
plt.title("Timezone offsets for sleep rows, removing |offset| > 12, removing spikes")


# if point has a neighbor that agrees, then we keep it
nowsleepcsv2 = []   
for i in range(1, len(nowsleepcsv)-1):
    last,cur,nxt = nowsleepcsv[i-1][1], nowsleepcsv[i][1], nowsleepcsv[i+1][1]
    if (last == cur) or (cur == nxt):
        nowsleepcsv2.append(copy.deepcopy(nowsleepcsv[i]))

tzOffs = [x[1] for x in nowsleepcsv2]
figure, ax = plt.subplots(figsize=(12,5))
ax.plot(tzOffs, marker=".")
ax.set_ylim([-12,12])
ax.hlines(0, 0, len(tzOffs), colors='r', linestyles='dashed')
plt.title(f"Timezone offsets for sleep rows, removing |offset| > 12, removing spikes, keeping doubles. Remaining rows: {len(nowsleepcsv2)}/{len(nowsleepcsv)}")


# #highlight x range
# ax.axvspan(220,240, alpha=0.5, color='red')

# If we see a spike at a transition, we amend it to the value on the left
# for i in range(1, len(nowsleepcsv)-1):
#     last,cur,next = nowsleepcsv[i-1][1], nowsleepcsv[i][1], nowsleepcsv[i+1][1]
#     if ((last < cur) and (cur > next) and (cur != next) and (cur != last)):
#         # Smooth the spike, make cur = last
#         nowsleepcsv[i][1] = last

# tzOffs = [x[1] for x in nowsleepcsv]
# figure, ax = plt.subplots(figsize=(12,5))
# ax.plot(tzOffs, marker=".")
# ax.set_ylim([-12,12])
# # ax.hlines(0, 0, len(tzOffs), colors='r', linestyles='dashed')


# Check for duplicates
a = [x[0] for x in nowsleepcsv2]
print(f"uniques in a: {len(set(a))}, total: {len(a)}")

# Take sleep df and adjust the times on days that we know the offset
for row in nowsleepcsv2:
    ind, hrOffset, ts, ts2 = row

    sleep.loc[ind, 'startTime'] = pd.to_datetime(sleep.loc[ind, 'startTime']) + pd.Timedelta(hours=hrOffset)
    sleep.loc[ind, 'endTime'] = pd.to_datetime(sleep.loc[ind, 'endTime']) + pd.Timedelta(hours=hrOffset)

newSleep = sleep.loc[[x[0] for x in nowsleepcsv2]]
newSleep.reset_index(drop=True, inplace=True)
newSleep['startTime'] = pd.to_datetime(newSleep['startTime'])
newSleep['endTime'] = pd.to_datetime(newSleep['endTime'])

# Save new sleep file
newSleep.to_csv("cooked/Sleep_sleep_utc.csv", index=False)


