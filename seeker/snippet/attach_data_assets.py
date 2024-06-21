#date: 2024-06-21T17:09:32Z
#url: https://api.github.com/gists/fdb25d30578f66bd1431959da82c786f
#owner: https://api.github.com/users/tmchartrand

import os
import yaml
import requests

with open("/root/capsule/environment/data_assets.yml", mode="r") as f:
    data = yaml.safe_load(f)
# in a running capsule, the CO_CAPSULE_ID is set as an environment variable
url = f'https://codeocean.allenneuraldynamics.org/api/v1/capsules/{os.getenv("CO_CAPSULE_ID")}/data_assets'
headers = {"Content-Type": "application/json"}
# my CO API key is attached to the capsule as API_SECRET
# if attached as a custom secret this might be CUSTOM_KEY instead
auth = "**********"
response = requests.post(url=url, headers=headers, auth=auth, json=data)
print(response.text)
a)
print(response.text)
