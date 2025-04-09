#date: 2025-04-09T16:51:31Z
#url: https://api.github.com/gists/5436df6cb33a964dfeec9e83cfe5f885
#owner: https://api.github.com/users/ryankelly-klaviyo

import csv
import sys

in_file = sys.argv[1]
in_file_two = sys.argv[2]
out_file = sys.argv[1].split(".csv")[0]+"_updated.csv"

profiles = {}

email_key = "Email Address"
preference_key = "True Value Swap"

for file in [in_file, in_file_two]:
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[email_key] in profiles:
                profiles[row[email_key]].append(row[preference_key])
            else:
                profiles[row[email_key]] = [row[preference_key]]

rows = [["Email", "Email Preferences"]]
for email, preferences in profiles.items():
    preference_string = "["
    pref_set = set(preferences)
    for preference in preferences:
        preference_string += f'"{preference}",'
    preference_string = preference_string[:-1]
    preference_string += "]"
    row = [email, preference_string]
    rows.append(row)

with open(out_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)