#date: 2022-05-02T17:02:10Z
#url: https://api.github.com/gists/027b3053c0275f01853f0877efdaaad3
#owner: https://api.github.com/users/m33ch33

#!/usr/bin/env python
# -*- coding: utf-8 -*-  
#
# This Hackvertor script shows a basic example of signing an HTTP body JSON message. 
# Each message has a timestamp and signature calculated by concatenating API-specific
# JSON keys altogether with a secret value/token and hashing them with SHA384. 
# The server upon message validation checks the timestamp and validates the signature
# by re-calculating the input JSON with the server-side stored authentication key of 
# the client. Also, the server decrypts the CVV with a symmetric key linked to the specific client.

import json
import hashlib

# Mock Params used for local testing
# apiName = "api1"
# clientSecret = "XXXXXXXXXXXXXXXXXXX"
# inputJSON = '{"field1":"John","field2":"Smith","field3":"authorization","field4":
# "USD","field5":1000,"wallet_data":{"email":"test@gmail.com","phone":"1111111111",
# "login":"333"},"field6":"378","locale":"en-GB","field7":"certainID","field8":
# "http:\/\/some.callback.url.com/callback","field9":"http:\/\/some.callback.url.com/callback2",
# "field10":"someAdditionalID","request_id":null,"variable1":"val1","variable2":"val2",
# "variable3":"val3","version":"1.3","field11":1631281107}'

#  Burp params
inputJSON = inJSON
clientSecret = str(mSecret)
apiName = str(api)

output = str(dir())
apiSignDic = {
    "api1": 
    ["field1","field2","field11","field3","field6","field10","field4","field5","field7","field8","field9"],
    "api2":
    ["field1","field2","field11","intent","field6","field10","agent_name"],
    "api3":
    ["field1","field2","field11","field3","field6","field10","field4","field5","field7","field8","field9"],
    "api4":
    ["field1","field2","field11","field3","field6","field10","field4","field5","field7","field8","field9","field13"],
    "api5":
    ["field1","field2","field11","field3","field6","field10","field4","field5","field7","field8","field9","field13"]}

jsonObj = json.loads(inputJSON)

# null values ignored
concat = ""
for k in apiSignDic[apiName]:
    for i in jsonObj:
        if i == k and jsonObj[i]:
            concat += str(jsonObj[i])  + "\n"

concat = concat.replace("\n","")
m = hashlib.sha384()
m.update(concat.encode('utf-8'))
hResult = m.hexdigest()
output = hResult

# output = concat
# print(concat)
# print("\n")
# print(hResult)
# print(apiSignDic[apiName][0])

# Burp raw example
# POST /controller/api HTTP/1.1
# Host: host.com
# Accept: */*
# Content-Type: application/json
# timestamp: <@set_ts('true')><@timestamp/><@/set_ts>
# Custom-Authentication: <@_APISign('api/name' ,'client-provided secret',
# 'Hackvertor code execution code')><@/_APISign>
# Content-Length: 0

# <@set_inJSON('false')>
# {
#     "field": "xxxx",
#     "field2": "xxxx",
#     "field3": "xxx",
#     "field4": "xxx",
#     "field5": 100,
#     "field6": "xxxxx",
#     "cvv_encrypted": "<@_EncryptCVV('client-provided secret','Hackvertor code execution code')>223<@/_EncryptCVV>",
#     "timestamp": <@get_ts/>
# }
# <@/set_inJSON>

# Burp result example
# POST /controller/api HTTP/1.1
# Host: host.com
# Accept: */*
# Content-Type: application/json
# timestamp: <@set_ts('true')><@timestamp/><@/set_ts>
# Custom-Authentication:0beedf0971f784a6635a14ae8edc4c000e697f714a273409838b2fc701678d526f5848a416e7da8f4e90525f5c56f7bd
# Content-Length: 0

# {
#     "field": "xxxx",
#     "field2": "xxxx",
#     "field3": "xxx",
#     "field4": "xxx",
#     "field5": 100,
#     "field6": "xxxxx",
#     "cvv_encrypted": "AESEncryptionResult",
#     "timestamp": 1631281107
# }

