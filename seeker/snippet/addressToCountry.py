#date: 2022-10-27T17:08:50Z
#url: https://api.github.com/gists/ffc1a958d2c27c6e78045ca8361a1ad8
#owner: https://api.github.com/users/godmingty

import json
import socket
from urllib.request import urlopen
while True:
    address = input('Enter address: ')
    ip_list = list({addr[-1][0] for addr in socket.getaddrinfo(address, 0, 0, 0, 0)})
    print(ip_list)
    for i in ip_list:
        response = urlopen(f"https://extreme-ip-lookup.com/json/{i}?key=LLRsYXbqHiFofI1E7UU5")
        geo = json.load(response)
        print(geo["country"])