#date: 2023-04-17T16:49:10Z
#url: https://api.github.com/gists/f3dea27e91acbb8414c3b9241c788f2e
#owner: https://api.github.com/users/AlexHenderson

# Expects a tab-delimited file with a column called "IP Address"
# Produces a version of the input file in the current folder with an additional column for the DNS name
# Doesn't seem to catch all DNS names, but useful anyway


import re
import subprocess

import pandas as pd

ipfilename = "my_ips.txt"
dnsfilename = "my_ips_and_dns.txt"

sheet = pd.read_csv(ipfilename, sep='\t')
sheet["DNS Name"] = ""
for index, row in sheet.iterrows():
    ip = row["IP Address"]
    response = subprocess.run(f"nslookup {ip}", capture_output=True)
    response = response.stdout.decode('utf-8')
    match = re.findall(r"Name\:\s+([\w+|\.|-]+)[\r\n|\r|\n]", response)

    if match:
        dns = ' '.join(match)
        sheet["DNS Name"][index] = dns

sheet.to_csv(dnsfilename, sep='\t', index=False)
print("done!")