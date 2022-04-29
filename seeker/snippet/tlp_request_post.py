#date: 2022-04-29T17:10:20Z
#url: https://api.github.com/gists/a5cab91be2f183a3f99c02bce74e25d2
#owner: https://api.github.com/users/ghoersti

import requests
import json
url = "https://stoplight.io/mocks/runtime/tlp-code-detection-api/40187599/tlp?apikey=''"
headers_ = {'Content-Type': 'application/json'}

code_to_eval = {'iterable': [
  {
   "value": "[i+i for i in range(10)]"
  }]
            }
x = requests.post(url, data = json.dumps(code_to_eval), headers = headers_ )
x.json()