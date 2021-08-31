#date: 2021-08-31T13:22:58Z
#url: https://api.github.com/gists/cf445530584b7574162cc1171d84d1f2
#owner: https://api.github.com/users/Jason-DataExplorer

import json
import requests

reservoir = []

url  = "https://www.taiwanstat.com/waters/latest"
re   = requests.get(url)
data = re.json()
data = data[0]

for d in data:
    reservoir.append(d)

for r in reservoir:
    name = data[r]["name"]
    volu = data[r]["volumn"]
    perc = data[r]["percentage"]
    print(f"名稱: {name}\n有效蓄水量: {volu}\n蓄水百分比: {perc}%")
    print("-------------------")