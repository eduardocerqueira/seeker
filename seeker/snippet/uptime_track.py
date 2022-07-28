#date: 2022-07-28T17:20:17Z
#url: https://api.github.com/gists/afb4ef74dce6a60d6d6b239be51db280
#owner: https://api.github.com/users/pulbee

import os
import math
import datetime

startStr = os.popen("uptime -s").read()

now = datetime.datetime.now()
startStrTime = datetime.datetime.strptime(startStr, '%d-%m-%Y %H:%M:%S')

nowStr = now.strftime("%d-%m-%Y %H:%M:%S")
secDiff = (now - startStrTime).total_seconds()
print("Diff : ", math.ceil(secDiff) )	