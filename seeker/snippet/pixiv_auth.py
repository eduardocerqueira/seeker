#date: 2025-05-09T17:09:34Z
#url: https://api.github.com/gists/008194435283c617b1bc0c029702b816
#owner: https://api.github.com/users/ILudw1g

#!/usr/bin/env python

from argparse import ArgumentParser
from base64 import urlsafe_b64encode
from hashlib import sha256
from pprint import pprint
from secrets import token_urlsafe
from sys import exit
from urllib.parse import urlencode
from webbrowser import open as open_url

import requests

# Latest app version can be found using GET /v1/application-info/android
USER_AGENT = "PixivAndroidApp/5.0.234 (Android 11; Pixel 5)"
REDIRECT_URI = "https://app-api.pixiv.net/web/v1/users/auth/pixiv/callback"
LOGIN_URL = "https://app-api.pixiv.net/web/v1/login"
AUTH_TOKEN_URL = "https: "**********"
CLIENT_ID = "MOBrBDS8blbauoSck0ZfDbtuzpyT"
CLIENT_SECRET = "**********"


def s256(data):
    """S256 transformation method."""

    return urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode("ascii")


def oauth_pkce(transform):
    """Proof Key for Code Exchange by OAuth Public Clients (RFC7636)."""

    code_verifier = "**********"
    code_challenge = transform(code_verifier.encode("ascii"))

    return code_verifier, code_challenge


 "**********"d "**********"e "**********"f "**********"  "**********"p "**********"r "**********"i "**********"n "**********"t "**********"_ "**********"a "**********"u "**********"t "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********"( "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********") "**********": "**********"
    data = response.json()

    try:
        access_token = "**********"
        refresh_token = "**********"
    except KeyError:
        print("error:")
        pprint(data)
        exit(1)

    print("access_token: "**********"
    print("refresh_token: "**********"
    print("expires_in:", data.get("expires_in", 0))


def login():
    code_verifier, code_challenge = oauth_pkce(s256)
    login_params = {
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "client": "pixiv-android",
    }

    open_url(f"{LOGIN_URL}?{urlencode(login_params)}")

    try:
        code = input("code: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    response = requests.post(
        AUTH_TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": "**********"
            "code": code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "include_policy": "true",
            "redirect_uri": REDIRECT_URI,
        },
        headers={"User-Agent": USER_AGENT},
    )

    print_auth_token_response(response)


 "**********"d "**********"e "**********"f "**********"  "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"( "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    response = requests.post(
        AUTH_TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": "**********"
            "grant_type": "**********"
            "include_policy": "true",
            "refresh_token": "**********"
        },
        headers={"User-Agent": USER_AGENT},
    )
    print_auth_token_response(response)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.set_defaults(func=lambda _: parser.print_usage())
    login_parser = subparsers.add_parser("login")
    login_parser.set_defaults(func=lambda _: login())
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("refresh_token")
    refresh_parser.set_defaults(func=lambda ns: "**********"
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
