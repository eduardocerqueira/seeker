#date: 2022-01-18T17:17:05Z
#url: https://api.github.com/gists/160ba5e1d5997c2213b5b0d9b2fc1670
#owner: https://api.github.com/users/balvinder294

#author @tekraze

import falcon
import requests

coalList = "https://gitlab.com/blurt/openblurt/coal/-/raw/master/coal.json"

class CoalList:
    def on_get(self, req, resp):

        response = requests.get(coalList)

        coal_json = response.json()

        response.close()

        jsonArr = []
        for key, value in coal_json.items():
            data = {}
            data['name'] = key

            data['reason'] = value['reason']
            data['notes']= value['notes']

            jsonArr.append(data)

        resp.media = jsonArr

middle = falcon.CORSMiddleware(
    allow_origins="*"
)

api = falcon.App(middleware=middle)

api.add_route('/', CoalList())
