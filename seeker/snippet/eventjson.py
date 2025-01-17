#date: 2025-01-17T17:12:26Z
#url: https://api.github.com/gists/905998809ed1b4277958eb512b07aa1f
#owner: https://api.github.com/users/jac18281828

import json

datastruct = json.loads(open("event/startnew.json").read())
datastruct = json.loads(datastruct)
print(json.dumps(datastruct, indent=4))
