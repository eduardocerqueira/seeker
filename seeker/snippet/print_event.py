#date: 2025-07-31T17:03:42Z
#url: https://api.github.com/gists/26e5e6de9b25db2be174ddff412ff4ad
#owner: https://api.github.com/users/mauriciosierrav

import json

def handler(event, context):
    print(json.dumps(event, default=str))
