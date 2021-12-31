#date: 2021-12-31T16:55:17Z
#url: https://api.github.com/gists/a9ca2bf07c3cf3d9fa6ca49b637cc686
#owner: https://api.github.com/users/Itz-fork

import requests

def search_ud(q):
  req = requests.get(f"https://nexa-apis.herokuapp.com/ud?query={q}").json()
  if req["status"] == "Ok":
      return req["data"]
  else:
    return None

def_txt = """
Definition: {}
Example: {}
Sounds (Urls): {}
Author: {}
Link to Urban Dictionary: {}

👍 {} : 👎 {}

---------------------------
"""

ud = search_ud("Bruh_0x")
if ud:
    for item in ud:
        defi = item["definition"]
        ex = item["example"]
        sounds = item["sounds"] if item["sounds"] else ""
        auth = item["author"]
        link = item["link"]
        likes = item["likes"]
        dislikes = item["dislikes"]
        print(def_txt.format(defi, ex, sounds, auth, link, likes, dislikes))
else:
    print("Oops, Error Happend!")