#date: 2021-11-10T17:09:07Z
#url: https://api.github.com/gists/4c17ed27b63400e6cb3469c1efe7e4b5
#owner: https://api.github.com/users/GottaBKD

import json

def lambda_handler(event, context):
    response = {}

    response["isBase64Encoded"]=False
    response["statusCode"]=302
    response["statusDescription"]="200 OK"

    if event["headers"].get("user-agent", None) == 'ELB-HealthChecker/2.0':
        response["statusCode"]=200
        return response
        
    stripped_host = event["headers"]["host"].removeprefix("www.")
    event["headers"]["host"] = stripped_host

    response["headers"] = {'Location': f"https://{stripped_host}"} | event["headers"]
    data = {}
    response["body"]=json.dumps(data)
    return response
