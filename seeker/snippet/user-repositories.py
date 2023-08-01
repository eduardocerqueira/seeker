#date: 2023-08-01T16:46:28Z
#url: https://api.github.com/gists/030a6aba3b0ff5fa36093a666c45e58b
#owner: https://api.github.com/users/Struchid

import requests

token = "**********"
username = "<your_github_username>"
headers = {"Authorization": "**********"
url = f"https://api.github.com/search/repositories?q=user:{username}"

response = requests.get(url, headers=headers)
print([item["full_name"] for item in response.json()["items"]])
()["items"]])
