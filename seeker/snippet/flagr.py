#date: 2021-11-09T17:18:41Z
#url: https://api.github.com/gists/aaae5bf24f390aaa03e43c8ddea3377c
#owner: https://api.github.com/users/vabshere

import requests
from os import environ
import base64
import json


envDict = dict(environ)

username = envDict.get("FLAGR_USERNAME")
password = envDict.get("FLAGR_PASSWORD")
host = envDict.get("FLAGR_HOST")
flagKey = envDict.get("FLAG_KEY")

def evaluate(requestParams):
    headers = {}
    creds = username + ":" + password
    headers['authorization'] = "Basic " + base64.b64encode(creds.encode('ascii')).decode('ascii')
    headers['content-type'] = 'application/json;charset=UTF-8'

    data = {}
    data['flagKey'] = flagKey
    data['entityContext'] = requestParams

    url = host + "/api/v1/evaluation"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.json().get('variantAttachment') is not None:
        return response.json().get('variantAttachment').get('message')

    return ""

def hello():
    message = "Hello!"
    requestParams = {}
    # requestParams = {"location": "ru"}
    
    res = evaluate(requestParams)
    if res != "":
        message = res

    print(message)

for i in range(10):
    hello()
