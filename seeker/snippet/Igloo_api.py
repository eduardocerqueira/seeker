#date: 2025-12-22T17:15:59Z
#url: https://api.github.com/gists/1145a15389e11dc60868ec28a50067a2
#owner: https://api.github.com/users/PeterTough2

//adapted for Zapier

import requests
import time

# -----------------------------
# INPUTS FROM ZAPIER
# -----------------------------
stored_token = "**********"
stored_expires_at = input_data.get("stored_expires_at")  # Unix timestamp (string)

DEVICE_ID = "xxxxxxxxxxxx"

# -----------------------------
# CONFIG
# -----------------------------
AUTH_URL = "https: "**********"
PIN_URL = f"https://api.igloodeveloper.co/igloohome/devices/{DEVICE_ID}/algopin/hourly"

AUTH_HEADER = "Basic xxxxxxxxxxx=="

SCOPES = (
    "igloohomeapi/algopin-permanent "
    "igloohomeapi/algopin-onetime "
    "igloohomeapi/algopin-hourly "
    "igloohomeapi/algopin-daily "
    "igloohomeapi/get-devices "
    "igloohomeapi/lock-bridge-proxied-job "
    "igloohomeapi/unlock-bridge-proxied-job"
)

NOW = int(time.time())

# -----------------------------
# STEP 1: "**********"
# -----------------------------
 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********") "**********": "**********"
    # Use cached token if still valid
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"t "**********"o "**********"r "**********"e "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"a "**********"n "**********"d "**********"  "**********"s "**********"t "**********"o "**********"r "**********"e "**********"d "**********"_ "**********"e "**********"x "**********"p "**********"i "**********"r "**********"e "**********"s "**********"_ "**********"a "**********"t "**********": "**********"
        try:
            if int(stored_expires_at) > NOW:
                return {
                    "access_token": "**********"
                    "expires_at": int(stored_expires_at),
                    "from_cache": True
                }
        except:
            pass

    # Otherwise login again
    response = requests.post(
        AUTH_URL,
        headers={
            "Authorization": AUTH_HEADER,
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "client_credentials",
            "scope": SCOPES
        },
        verify=False,
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    expires_at = NOW + int(data.get("expires_in", 86400)) - 60  # buffer

    return {
        "access_token": "**********"
        "expires_at": expires_at,
        "from_cache": False
    }

# -----------------------------
# STEP 2: GENERATE PIN
# -----------------------------
 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"p "**********"i "**********"n "**********"( "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    response = requests.post(
        PIN_URL,
        headers={
            "Authorization": "**********"
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        json={
            "variance": 1,
            # "startDate": input_data.get("startDate"),
            "startDate": "2025-12-25T12:00:00+01:00",
            "endDate": "2025-12-25T16:00:00+01:00",
            # "endDate": input_data.get("endDate"),
            "accessName": "Testing zaps Maintenance guy test"
        },
        verify=False,
        timeout=30
    )

    response.raise_for_status()
    return response.json()

# -----------------------------
# MAIN FLOW
# -----------------------------
token_info = "**********"
pin_data = "**********"

# -----------------------------
# RETURN STRUCTURED DATA
# -----------------------------
return {
    # Token info (store these back into Storage by Zapier)
    "access_token": "**********"
    "token_expires_at": "**********"
    "token_from_cache": "**********"

    # PIN info (for Gmail)
    "pin": pin_data.get("pin"),
    "pin_id": pin_data.get("pinId"),

    # Status
    "success": True
}