#date: 2023-03-01T16:42:03Z
#url: https://api.github.com/gists/31279af16946fc94df8384af5514c33c
#owner: https://api.github.com/users/nfl0

import requests

url = "https://rpc-mumbai.maticvigil.com"

try:
    response = requests.post(url, json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1})
    if response.status_code == 200 and "result" in response.json():
        print("RPC is operational.")
        print(response.json())
    else:
        print("RPC is not operational.")
except:
    print("RPC is not operational.")
