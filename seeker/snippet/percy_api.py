#date: 2022-11-07T17:04:29Z
#url: https://api.github.com/gists/2dbeb4ae9aae046551f831ec2ff9fbe7
#owner: https://api.github.com/users/ansel2000

import requests
import json
import sys
import os

url = "https://percy.io/api/v1/projects?project_slug=%s"%(sys.argv[1])

headers = {"Authorization": "**********"
response = requests.get(url, headers=headers)
response = response.json()
project_id = response["data"]["id"]

url = "https://percy.io/api/v1/builds?project_id=%s"%(project_id)

response = requests.get(url, headers=headers)
response = response.json()
build_id = response['data'][0]['id']

url = "https://percy.io/api/v1/builds/%s"%(build_id)

response = requests.get(url, headers=headers)
response = response.json()
build_state = response['data']['attributes']['state']
build_review_state = response['data']['attributes']['review-state']
build_review_state_reason = response['data']['attributes']['review-state-reason']
print("Below is the status of the build %s\n"%(build_id))
print("state: ",build_state)
print("review-state: ",build_review_state)
print("review-state-reason: ",build_review_state_reason)
print("\n====================================================\n")

url = "https://percy.io/api/v1/snapshots?build_id=%s"%(build_id)
response = requests.get(url, headers=headers)
response = response.json()
print("\nBelow is the status of the snapshot for build %s\n"%(build_id))
for snapshot in response['data']:
    snapshot_name = snapshot['attributes']['name']
    snapshot_review_state = snapshot['attributes']['review-state']
    snapshot_review_state_reason = snapshot['attributes']['review-state-reason']
    if snapshot_review_state != "approved":
        print("name: ",snapshot_name)
        print("review-state: ",snapshot_review_state)
        print("review-state-reason: ",snapshot_review_state_reason)
        print('\n')reason)
        print('\n')