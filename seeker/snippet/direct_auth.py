#date: 2023-02-01T16:59:32Z
#url: https://api.github.com/gists/9f58aee7697a1d0756125ac93b5a27e8
#owner: https://api.github.com/users/ffenix113

#!/usr/bin/env python3
import os
import sys

import requests

###
### Configuration starts here

CLIENT_ID = 'client_id'
CLIENT_SECRET = "**********"

# This should be in form of 
# Keycloak <17: https://example.com/auth/realms/<realm_name>
# Keycloak >=17: https://example.com/realms/<realm_name>
REALM_URL = 'https://auth.mega.pp.ua/realms/mega'

### Configuration ends here
###

###
### Some functions to modify behavior

def get_meta_attributes(token: "**********":
    """ 
    See https://www.home-assistant.io/docs/authentication/providers/#command-line
    """
    name = "**********"
    if not name:
        name = "**********"

    return {
        'name': name
    }
 
def login_hook(username: "**********": dict, token: dict) -> bool:
    """ 
    This function will decide if user can access HA or not.
    """
    return 'home_assistant' in token['realm_access']['roles']

### End functions to modify behavior
###

# Do not change this
AUTH_URL = "**********"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def decode_token(token: "**********":
    import json
    from base64 import b64decode

    access_token = "**********"
    # Some nice fix for invalid padding
    # https://gist.github.com/perrygeo/ee7c65bb1541ff6ac770
    decoded = "**********"===").decode('utf8')
    return json.loads(decoded)

def auth(username: "**********": str) -> bool:
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': "**********"
        'username': username,
        'password': "**********"
        'grant_type': "**********"
    }
    resp = requests.post(AUTH_URL, data=payload)
    if resp.status_code != 200:
        eprint(f'wrong status code: {resp.status_code}, response: {resp.content}')
        return False

    data = resp.json()

    decoded_token = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********"_ "**********"h "**********"o "**********"o "**********"k "**********"( "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"d "**********"a "**********"t "**********"a "**********", "**********"  "**********"d "**********"e "**********"c "**********"o "**********"d "**********"e "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        return False

    attributes = "**********"
    for attribute in attributes:
        print(f'{attribute} = {attributes[attribute]}')

    return True


if __name__ == '__main__':
    username = os.getenv('username')
    password = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"u "**********"t "**********"h "**********"( "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
        exit(1)