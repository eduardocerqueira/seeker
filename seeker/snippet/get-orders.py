#date: 2021-11-23T17:04:51Z
#url: https://api.github.com/gists/0f021e690a6c5bd211c267309b2fe31a
#owner: https://api.github.com/users/mariussprenger

import requests

request = requests.get("https://paper-trading.lemon.markets/v1/orders/?isin=US88160R1014&&side=buy", 
                       headers={"Authorization": "Bearer YOUR-API-KEY"})
print(request.json())