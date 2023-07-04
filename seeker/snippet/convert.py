#date: 2023-07-04T17:07:26Z
#url: https://api.github.com/gists/1c2760b53f008d24e9a769ef28d896d8
#owner: https://api.github.com/users/krzysztoff1

import json
import csv

json_file = "users.json"

with open(json_file) as file:
    json_data = file.read()

data = json.loads(json_data)

users = data["users"]

csv_file = "user_data.csv"

fieldnames = ["Email address"];

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    for user in users:
        email = user["email"]

        writer.writerow({
            "Email address": email,
        })

print(f"Data converted and saved to {csv_file}")
