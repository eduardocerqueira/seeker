#date: 2022-06-17T16:47:03Z
#url: https://api.github.com/gists/557306aa74a40ef1ec5a75856e54aa74
#owner: https://api.github.com/users/StevenTso

import requests

# get price in USD
url = 'https://api.metalpriceapi.com/v1/latest?api_key=[API_KEY]&base=USD&currencies=XAU'
r = requests.get(url)
gold_dict = r.json()
rate = gold_dict["rates"]["XAU"]
gold_price = 1/rate
print(gold_price)

# get price is MXN
url = 'https://api.metalpriceapi.com/v1/latest?api_key=[API_KEY]&base=MXN&currencies=XAU'
r = requests.get(url)
gold_dict = r.json()
rate = gold_dict["rates"]["XAU"]
gold_price = 1/rate
print(gold_price)