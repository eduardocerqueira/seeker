#date: 2021-12-13T17:14:09Z
#url: https://api.github.com/gists/296c392a15f1ffd242d8ff09e9ae22a5
#owner: https://api.github.com/users/mariussprenger

import requests

request = requests.get("https://paper-trading.lemon.markets/v1/portfolio/?isin=US88160R1014", 
                       headers={"Authorization": "Bearer YOUR-API-KEY"})
print(request.json())