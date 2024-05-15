#date: 2024-05-15T17:09:04Z
#url: https://api.github.com/gists/4616de7004fb9306814fe8236c27382e
#owner: https://api.github.com/users/hn4002


# This script is used to manually authenticate the user with the Schwab API.
# It deletes the token file at the beginning. It also creates a meta file with the current timestamp.

import datetime
import json
import os
import pathlib
import pytz
import sys

import schwab

from environment.instance_settings import schwabSettings

client_id = schwabSettings.SCHWAB_APP_ID
client_secret = "**********"
redirect_uri = schwabSettings.SCHWAB_REDIRECT_URI
token_path = "**********"

#===============================================================================
def manual_auth():
    # First delete the token file if it exists
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"o "**********"s "**********". "**********"p "**********"a "**********"t "**********"h "**********". "**********"e "**********"x "**********"i "**********"s "**********"t "**********"s "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"p "**********"a "**********"t "**********"h "**********") "**********": "**********"
        os.remove(token_path)

    client = schwab.auth.client_from_manual_flow(
        api_key=client_id,
        app_secret= "**********"
        callback_url=redirect_uri,
         "**********"= "**********"
    )

    # Create a file to save the timestamp for the last time the refresh token was generated
    meta_path = "**********"
    # First delete the meta file if it exists
    if os.path.exists(meta_path):
        os.remove(meta_path)
    # Now create the meta file with the current timestamp
    curr_dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
    data = {
        'last_refresh_time': curr_dt.isoformat()
    }
    with open(meta_path, "w") as f:
        f.write(json.dumps(data, indent=4))


#===============================================================================
if __name__ == '__main__':
    manual_auth()
    
    