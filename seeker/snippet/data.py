#date: 2023-09-27T17:01:11Z
#url: https://api.github.com/gists/f45cd57f77c5e2fd181cc847a2cb496c
#owner: https://api.github.com/users/Phil-Miles

import requests

parameters = {
    "amount": 10,
    "type": "boolean",
}

response = requests.get("https://opentdb.com/api.php", params=parameters)
response.raise_for_status()
data = response.json()
question_data = data["results"]
print(question_data)
