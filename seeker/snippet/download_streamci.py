#date: 2025-08-06T17:08:41Z
#url: https://api.github.com/gists/2d261ded65abf9edebb80909bc6f5ff2
#owner: https://api.github.com/users/jakehosen

import requests
import json
import sys
import datetime
import pandas as pd


 "**********"d "**********"e "**********"f "**********"  "**********"f "**********"e "**********"t "**********"c "**********"h "**********"( "**********"q "**********"u "**********"e "**********"r "**********"y "**********", "**********"  "**********"t "**********"a "**********"r "**********"g "**********"e "**********"t "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"k "**********"e "**********"y "**********") "**********": "**********"
    headers = {'Content-type': 'application/json'}
    data = {
            "auth": {
                "target":target,
                "authtype": "**********"
                " "**********"": "**********"
                },
            "request": query
    }
    data["request"]["batch"] = 0

    ret = []
    while True:
        addr = 'https://api.streamci.org/query'
        res = requests.post(addr, data=json.dumps(data), headers=headers)
        try:
            result = json.loads(res.content)
        except:
            print("ERROR")
            print(res.content)
            break
        if "data" in result:
            print("batch: " + str(data["request"]["batch"]) + ": " +str(len(result["data"])) + " data")
        else:
            print(result)
        ret += result["data"]

    return ret


# # aqsensors
target = "aqsensors"
secret_key = "**********"
query = {
    "method":"query",
    "query":{}
    # ,"projection":["value", "sequence", "timestamp"]
#    ,"sort":{"time_h":1}
}





# Get objects within fifteen minutes
edt = datetime.datetime.utcnow()
sdt = edt - datetime.timedelta(minutes=15)
sdt = sdt.isoformat()[:-3] + "Z"
edt = edt.isoformat()[:-3] + "Z"
query["query"] = {"published_at": {"$gte": sdt, "$lt": edt}}
results = "**********"
print(query)
print(target)
print(secret_key)
print(results)

