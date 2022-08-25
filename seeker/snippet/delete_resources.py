#date: 2022-08-25T15:26:52Z
#url: https://api.github.com/gists/1519cf80cef1d75af44db586117e7243
#owner: https://api.github.com/users/joofio

import requests

RESOURCES_TO_DELETE = ["Medication", "MedicationKnowledge"]
URL_BASE = "https://server.fire.ly/r4/"

for resource in RESOURCES_TO_DELETE:
    url = URL_BASE + resource
    data = requests.get(url)
    # print(data.json())
    if "entry" in data.json().keys():
        for entry in data.json()["entry"]:
            print(entry)
            r = requests.delete(url + "/" + entry["resource"]["id"])
            print(r.status_code)
        # print(r.text)
    #   print()
    else:
        print("No entries found for {}".format(resource))
