#date: 2025-04-02T17:09:04Z
#url: https://api.github.com/gists/89648eb2cdd7233b1267553284c7a22d
#owner: https://api.github.com/users/Luc-Mcgrady

import json
from statistics import mean

new_file = "result/FSRS-5-dev.jsonl"
old_file = "result/FSRS-5.jsonl"

#file1 = "result/FSRS-5-dev-active2.jsonl"
#file2 = "result/FSRS-5-dev.jsonl"

with open(new_file, "r") as f:
    new = [json.loads(x) for x in f.readlines()]
all_new = { x["user"]: x for x in new}
new = all_new # {x["user"]: x for x in new if x["parameters"]["0"][15] < 0}


with open(old_file, "r") as f:
    old = [json.loads(x) for x in f.readlines()]
old = {x["user"]: x for x in old}

intersection = all_new.keys() & old.keys()

def display(results):
    results = sorted(results, key=lambda user: abs(old[user]["metrics"]["LogLoss"]-new[user]["metrics"]["LogLoss"]))
    for user in results:
        before = old[user]["metrics"]["LogLoss"]
        after = new[user]["metrics"]["LogLoss"]
        print(f"{user=} w15:{new[user]['parameters']['0'][15]}, {before=}, {after=}, {before-after=}")

def display_header(users, title):
    values = [old[user]["metrics"]["LogLoss"] - new[user]["metrics"]["LogLoss"] for user in users]
    print(f"---- {title}: {len(values)=} {mean(values)=:} {sum(values)=} ----")

# Worsening of log loss
worse = [user for user in intersection if old[user]["metrics"]["LogLoss"] < new[user]["metrics"]["LogLoss"]]
display_header(worse, "Worse")
display(worse)
# Improvement in log loss
better = [user for user in intersection if old[user]["metrics"]["LogLoss"] > new[user]["metrics"]["LogLoss"]]
display_header(better, "Better")
display(better)
# Improvement in log loss and a negative w[15]
abusers = [user for user in intersection if old[user]["metrics"]["LogLoss"] > new[user]["metrics"]["LogLoss"] and new[user]['parameters']['0'][15] < 0]
display_header(abusers, "Abusers")
display(abusers)

# display_header(better + worse, "Total -w[15]")
display_header(all_new.keys() & old.keys(), "Total")