#date: 2023-09-05T16:56:14Z
#url: https://api.github.com/gists/387e9a5b6dc78a36931055991b938662
#owner: https://api.github.com/users/rudenskagjini

#!/usr/bin/env python3
# Mark Baggett @MarkBaggett graciously wrote this script.
# Minor changes by Joshua Wright @joswr1ght.
# Use it to retrieve host name information from the JSON output of tls-scan
# (https://github.com/prbinu/tls-scan) in the subjectCN and subjectAltName
# fields.

import json
import re
import sys
import pdb
def filter_hostnames(unfiltered):
    if re.match(r"kubernetes|kube-api|ip-.*internal",unfiltered):
        return None
    filtered = unfiltered.replace("DNS:","").replace("IP Address:","").replace("*.","")
    return filtered

if (len(sys.argv) != 2):
    print("Extract host name information from TLS-Scan JSON certificate details.")
    print("This isn't perfect, and you will likely need to do some manual filtering of these results.\n")
    print(f"Usage: {sys.argv[0]} <tls-scan-output.json>")
    sys.exit(0)
with open(sys.argv[1], "rb") as fc:
    data = fc.readlines()
    certsubjects = []
    for each_rec in data:
        json_rec = json.loads(each_rec)
        cert_chain = json_rec.get("certificateChain",[])
        for each_cert in cert_chain:
            subject = each_cert.get("subjectCN","")
            subject = filter_hostnames(subject)
            # Only include entries that do not include a space and at least one dot as a test for hostname viability
            if subject and " " not in subject and "." in subject:
                certsubjects.append(json_rec["ip"] + ":" + subject)
            altsubject  = each_cert.get("subjectAltName","")
            altsubject = filter_hostnames(altsubject)
            if altsubject:
                # Subject entries may be a comma-seperated string of values
                subjects = altsubject.split(", ")
                certsubjects = certsubjects + [ json_rec["ip"] + ":" + x for x in subjects if " " not in x and "." in x ]
    for each_subject in sorted(set(certsubjects)):
        print(each_subject)