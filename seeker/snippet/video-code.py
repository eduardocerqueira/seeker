#date: 2024-06-20T16:59:39Z
#url: https://api.github.com/gists/fdf60454aa674d999862796b2b32a9f4
#owner: https://api.github.com/users/ChadThackray

import requests
import json

bob = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=gbp", headers = {"accept":"application/json"})

print("The price of bitcoin is currently £" + str(bob.json()["bitcoin"]["gbp"]))