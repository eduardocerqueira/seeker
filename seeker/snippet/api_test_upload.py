#date: 2023-03-14T17:06:11Z
#url: https://api.github.com/gists/d21d1c8666fb42fda807b4899d98f8cf
#owner: https://api.github.com/users/aecgabriel

import requests
import json
import urllib.parse as url

#Call Weduu API to return the OAuth 2.0 token to connect to GCP API.
wdu_response = json.loads(requests.api.post(url='https: "**********"
                             json={
                                'key':'67cca30d-6ef1-4015-a42a-715a18175e94'
                             }).text)

access_token = "**********"

object_location = 'test.txt'
bucket_name = 'dev-wdu-sba-kcc_pe'
file_name = url.quote("tmp/test.txt")
url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={file_name}"
headers= {'Authorization': "**********"

with open(object_location, "rb") as f:
    #Call GCP API to upload a file
    gcp_response = requests.post(url, headers=headers, data=f)
 a file
    gcp_response = requests.post(url, headers=headers, data=f)
