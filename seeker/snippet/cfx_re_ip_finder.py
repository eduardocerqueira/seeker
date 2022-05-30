#date: 2022-05-30T17:00:34Z
#url: https://api.github.com/gists/1706b0c7c603215b63e6d61bbb904f88
#owner: https://api.github.com/users/HirboSH

import requests

cfx_re_link = str(input(">>> Input your server join link: "))
cfx_re_link = cfx_re_link if cfx_re_link.startswith("https://") else f"https://{cfx_re_link}" 

try:
    r = requests.get(cfx_re_link)

    ip = f">>> Server IP: {((r.headers['X-Citizenfx-Url'][7:]).replace('/', ''))}" if r.status_code == requests.codes.ok else ">>> Server is either offline or do not exists."

    print(ip)
except requests.exceptions.RequestException as e:
    print(">>> Server is either offline or do not exists.")