#date: 2024-01-23T17:01:41Z
#url: https://api.github.com/gists/ff6cbb42b77fc44c422e271ab5d4b76d
#owner: https://api.github.com/users/akanchhaS

import requests

SNYK_TOKEN = "**********"

# API endpoint url
url = "https://api.snyk.io/rest/orgs?version=2023-08-29&limit=100"

headers = {
    'Accept': 'application/vnd.api+json',
    'Authorization': "**********"
}

# make request here
res = requests.request("GET", url, headers=headers)

# process response here
print(res)eaders=headers)

# process response here
print(res)