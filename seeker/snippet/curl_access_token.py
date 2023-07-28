#date: 2023-07-28T16:57:23Z
#url: https://api.github.com/gists/ab1eea8fab3d32c83e23b8ba82e540d4
#owner: https://api.github.com/users/mario0986

import time, json

# pip3 install google-auth
import google.auth.crypt
import google.auth.jwt

# pip3 install urllib3
import urllib.parse


# OVERALL SCRIPT SUMMARY
"""This script generates the curl to get the access token from google"""

# CONSIDERATIONS
SCOPE = "https://www.googleapis.com/auth/devstorage.full_control"
# The scope you want to give access
EXPIRY_LENGTH = 3600
# The lenght of the token
SA_FILE_NAME = "service_account.json"
# The service account file name
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
# The grant type to use JWT


def generate_jwt(sa_file_name, sa_email, audience, scope, expiry_length):
    """Generates a signed JSON Web Token using a Google API Service Account."""

    now = int(time.time())
    # build payload
    payload = {
        "iat": now,
        # expires after 'expiry_length' seconds.
        "scope": scope,
        "exp": now + expiry_length,
        # iss must match 'issuer' in the security configuration in your
        # swagger spec (e.g. service account email). It can be any string.
        "iss": sa_email,
        # aud must be either your Endpoints service name, or match the value
        # specified as the 'x-google-audience' in the OpenAPI document.
        "aud": audience,
        # sub and email should match the service account's email address
        "sub": sa_email,
        "email": sa_email,
    }

    # payload_object = json.loads(payload)
    # print("Payload {payload_object}")

    # sign with keyfile
    signer = google.auth.crypt.RSASigner.from_service_account_file(sa_file_name)
    jwt = google.auth.jwt.encode(signer, payload).decode()
    return jwt


def generate_curl(jwt, audience):
    encoded_grant_type = urllib.parse.quote(GRANT_TYPE)
    curl_string = (
        f"curl -d 'grant_type={encoded_grant_type}&assertion={jwt} ' {audience}"
    )
    return curl_string


if __name__ == "__main__":
    with open(SA_FILE_NAME, "r") as f:
        data = json.load(f)

    # Now the data variable contains the data from the JSON file
    sa_email = data["client_email"]
    audience = "**********"
    expiry_length = EXPIRY_LENGTH

    jwt = generate_jwt(SA_FILE_NAME, sa_email, audience, SCOPE, expiry_length)
    curl = generate_curl(jwt, audience)
    print(curl)
url)
