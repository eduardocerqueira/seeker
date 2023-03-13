#date: 2023-03-13T16:54:08Z
#url: https://api.github.com/gists/c62c3afe2d621b6f586fff130828f25b
#owner: https://api.github.com/users/FilipRazek

import os

OAUTH_REFRESH_TOKEN = "**********"
OAUTH_CLIENT_ID = os.environ.get('OAUTH_CLIENT_ID')
OAUTH_CLIENT_SECRET = "**********"

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********") "**********": "**********"
    refresh_request_body = {
        "grant_type": "**********"
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": "**********"
        "refresh_token": "**********"
    }
    response = requests.post("https: "**********"
    return response.json()['access_token']