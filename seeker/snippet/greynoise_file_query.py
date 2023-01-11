#date: 2023-01-11T16:49:57Z
#url: https://api.github.com/gists/139b6658b7cf2e8f66972bf7f94f5ec7
#owner: https://api.github.com/users/superducktoes

import fileinput
import re
from greynoise import GreyNoise

# command usage: cat <file_ips>.txt| python3 file_ips_lookup.py

api_client = GreyNoise(api_key="YOUR_API_KEY")
ip_list = []

# Parse lines of file into array
for line in fileinput.input():
    ip = re.findall( r'[0-9]+(?:\.[0-9]+){3}', line )
    if ip:
        for i in ip:
            ip_list.append(i)

# post the results to GreyNoise
greynoise_results = api_client.quick(ip_list)

for i in greynoise_results:
    print("IP: {} - Noise Status: {}".format(i["ip"], i["noise"]))