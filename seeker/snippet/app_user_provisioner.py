#date: 2022-07-20T17:19:26Z
#url: https://api.github.com/gists/d538a40901aad8e5057bf0aeb8081ea6
#owner: https://api.github.com/users/lognaturel

import base64
import json
import glob
import requests
import segno
from typing import Optional
import zlib
from PIL import Image, ImageOps, ImageFont, ImageDraw

# Put a series of names in a file named users.csv in this directory to create their App Users on a server as configured below.

SERVER_URL = "https://test.getodk.cloud"
PROJECT = 149
PROJECT_NAME = "Cool Project"
USERNAME = ""
PASSWORD = ""

FORMS_TO_ACCESS = ["all-widgets", "essn-100-repeats"]
APP_USER_ROLE_ID = 2

COLLECT_SETTINGS = { 
  "general": {
    "form_update_mode": "match_exactly",
    "autosend": "wifi_and_cellular",
    "delete_send": True
  },
  "admin": {
 	"admin_pw": "s00p3rs3cr3t",
    "change_server": False,
    "automatic_update": False,
    "change_autosend": False
  },
  "project": {
      "color": "#ffeb3b",
      "icon": "💥"
  }
}

def get_token(url: str, username: str, password: str, cache_file: Optional[str] = None):
    """Get a verified session token with the provided credential. First tries from cache if a cache file is provided,
    then falls back to requesting a new session"
    Parameters:
    url: the base URL of the Central server to connect to
    username: the username of the Web User to auth with
    password: the Web User's password
    cache_file (optional): a file for caching the session token. This is recommended to minimize the login events logged
        on the server.
    Returns:
    Optional[str]: the session token or None if anything has gone wrong
    """
    token = get_verified_cached_token(url, cache_file) or get_new_token(url, username, password)
    if not token:
        raise SystemExit("Unable to get session token")

    if cache_file is not None:
        write_to_cache(cache_file, "token", token)

    return token


def get_verified_cached_token(url: str, cache_file: Optional[str] = None) -> Optional[str]:
    """Try to read a Central session token from the "token" property of a JSON cache file with the given filename"""
    if cache_file is None:
        return None

    try:
        with open(cache_file) as cache_file:
            cache = json.load(cache_file)
            token = cache["token"]
            user_details_response = requests.get(
                f"{url}/v1/users/current",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
            )
            if user_details_response.ok:
                return token
    except (FileNotFoundError, KeyError):
        return None


def get_new_token(url: str, username: str, password: str) -> Optional[str]:
    """Get a new token from Central by creating a new session (https://odkcentral.docs.apiary.io/#reference/authentication/session-authentication/logging-in)
    Parameters:
    url: the base URL of the Central server to connect to
    username: the username of the Web User to auth with
    password: the Web User's password
    Returns:
    Optional[str]: the session token or None if anything has gone wrong
    """
    email_token_response = requests.post(
        f"{url}/v1/sessions",
        data=json.dumps({"email": username, "password": password}),
        headers={"Content-Type": "application/json"},
    )

    if email_token_response.status_code == 200:
        return email_token_response.json()["token"]


def write_to_cache(cache_file: str, key: str, value: str):
    """Add the given key/value pair to the provided cache file, preserving any other properties it may have"""
    try:
        with open(cache_file) as file:
            cache = json.load(file)
            cache[key] = value
    except FileNotFoundError:
        cache = {key: value}

    with open(cache_file, 'w') as outfile:
        json.dump(cache, outfile)


with open('users.csv', newline='') as f:
    desired_users = f.readlines()
    desired_users = [user.rstrip() for user in desired_users]

token = get_token(SERVER_URL, USERNAME, PASSWORD, "cache.json")

current_users = requests.get(f"{SERVER_URL}/v1/projects/{PROJECT}/app-users",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    ).json()
current_users = [user for user in current_users if user["token"] is not None]

current_display_names = [user['displayName'] for user in current_users]
to_provision = set(desired_users) - set(current_display_names)

for user in to_provision:
    print("Provisioning: " + user)
    try:
        provision_request = requests.post(f"{SERVER_URL}/v1/projects/{PROJECT}/app-users",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            data="{\"displayName\": \"" + user + "\"}")
        provision_request.raise_for_status()
    except Exception as err:
        print(err)

    user_id = provision_request.json()['id']

    for formid in FORMS_TO_ACCESS:
        try:    
            assignment_request = requests.post(f"{SERVER_URL}/v1/projects/{PROJECT}/forms/{formid}/assignments/{APP_USER_ROLE_ID}/{user_id}",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
            )
            provision_request.raise_for_status()
        except Exception as err:
            print(err)

    url = f"{SERVER_URL}/v1/key/{provision_request.json()['token']}/projects/{PROJECT}"
    COLLECT_SETTINGS["general"]["server_url"] = url
    COLLECT_SETTINGS["project"]["name"] = f"{PROJECT_NAME}: {user}"
    COLLECT_SETTINGS["general"]["username"] = user
    
    qr_data = base64.b64encode(zlib.compress(json.dumps(COLLECT_SETTINGS).encode("utf-8")))

    code = segno.make(qr_data, micro=False)
    code.save("settings.png", scale=4)

    png = Image.open('settings.png')
    png = png.convert('RGB')
    text_anchor = png.height
    png = ImageOps.expand(png, border=(0, 0, 0, 30), fill = (255, 255, 255))
    draw = ImageDraw.Draw(png)
    font = ImageFont.truetype("Roboto-Regular.ttf", 24)
    draw.text((20, text_anchor - 10), user, font = font, fill = (0, 0, 0))
    png.save(f"settings-{user}.png", format = 'PNG')

images = [Image.open(f) for f in glob.glob('./settings-*.png')]

images[0].save("pdf", "PDF", resolution=100, save_all=True, append_images=images[1:])
