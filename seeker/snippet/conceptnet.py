#date: 2023-11-03T16:49:32Z
#url: https://api.github.com/gists/a99045a46b0a9e2184d7f10ee8b7ae12
#owner: https://api.github.com/users/ToroData

import requests

word = "computer_security"
response = requests.get(f"http://api.conceptnet.io/c/en/{word}")
data = response.json()
for edge in data["edges"]:
    print(edge["start"]["label"], "-", edge["rel"]["label"], "-", edge["end"]["label"])
