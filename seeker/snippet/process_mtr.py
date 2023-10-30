#date: 2023-10-30T16:47:39Z
#url: https://api.github.com/gists/aa1fef4075890bc59785e688b9a02976
#owner: https://api.github.com/users/MTN-RowinAndruscavage

#!python

# Convert mtr json logs to a csv report for graphing
# Collect data intermittently from mtr with:
# while true; do mtr -j 8.8.8.8 > mtr_$(date +'%Y-%m-%d_%H:%M:%S').json; sleep 30; done

import os,datetime
import glob
import json

filenames = glob.glob('./*.json')

output = open('summary.csv', 'w')
output.write("time, min, avg, max, stdev, loss\n")

for fn in sorted(filenames):
    print(fn)
    try:
        with open(fn) as fd:
            data = json.load(fd)
    except:
        print("Skipping ", fn)
        pass


    ts = fn[6:-5]
    # print(ts)

    if not 'report' in data:
        pass

    try:
        di = next(item for item in data['report']['hubs'] if item["host"] == "dns.google")
    except:
        print("dns.google not found; skipping")
        pass

    # print(data['report']['hubs'][0]['Avg']) # .report.hubs[0].host)

    line = f"{ts}, {di['Best']}, {di['Avg']}, {di['Wrst']}, {di['StDev']}, {di['Loss%']}"

    print(line)
    output.write(line + '\n')

output.close()