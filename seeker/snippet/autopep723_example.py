#date: 2025-07-22T16:53:00Z
#url: https://api.github.com/gists/8fb70482b46454cc22cb6f2417afb8ea
#owner: https://api.github.com/users/mgaitan

import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])