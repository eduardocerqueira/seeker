#date: 2023-08-01T16:43:23Z
#url: https://api.github.com/gists/42424ae3a67b55d5c9f67d51779bfb62
#owner: https://api.github.com/users/Struchid

import requests

username = "<your_github_username>"
api_url = f"https://api.github.com/users/{username}"
response = requests.get(api_url)
print(response.status_code)
print(response.json())
