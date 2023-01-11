#date: 2023-01-11T17:06:24Z
#url: https://api.github.com/gists/bf8241b38cb6a4ade50a4f5c2b7937b3
#owner: https://api.github.com/users/dwedigital

import json
import csv

data = {}
# Opening JSON file
with open("applicants.json") as json_file:
    data = json.load(json_file)
applicants = []
for candidate in data["data"]:
    applicants.append(
        {
            "candidate_id": candidate["id"],
            "job_id": candidate["job_id"],
            "candidate_name": candidate["full_name"],
            "channel": candidate["channel"],
            "how they heard": candidate["answers"][3]["multiple_choice_answer"],
        }
    )

with open("applicants.csv", "w", newline="") as csvfile:
    fieldnames = [
        "candidate_id",
        "job_id",
        "candidate_name",
        "channel",
        "how they heard",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for applicant in applicants:
        writer.writerow(applicant)
