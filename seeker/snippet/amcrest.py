#date: 2024-12-02T17:11:07Z
#url: https://api.github.com/gists/be7cff34e2ccdd8fda90637aea05fc24
#owner: https://api.github.com/users/adamkornafeld

import requests
import hashlib
import time
import logging

# Constants
LOGIN_METHOD = "global.login"
LOGOUT_METHOD = "global.logout"
CONTROL_METHOD = "CoaxialControlIO.control"
CLIENT_TYPE = "Web3.0"
AUTHORITY_TYPE = "Default"
ENCRYPTION_TYPE = "Default"
IO_TYPE_ON = 1
IO_TYPE_OFF = 2

# Configuration
id = 1
user = "admin"
password = "**********"
camera_ip = "CAMERA_IP_HERE"

loginURL = f"http://{camera_ip}/RPC2_Login"
rpcURL = f"http://{camera_ip}/RPC2"

# Setup logging
logging.basicConfig(level=logging.INFO)

def make_request(url, payload):
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        logging.error(f"Request failed with status code: {response.status_code}")
        raise Exception("Request failed")
    return response.json()

def login():
    global id
    first = {
        "method": LOGIN_METHOD,
        "params": "**********": user, "password": "", "clientType": CLIENT_TYPE, "loginType": "Direct"},
        "id": id
    }
    resp = make_request(loginURL, first)
    session = resp['session']
    params = resp['params']
    random = params['random']
    realm = params['realm']
    encryption = params['encryption']

    if encryption != ENCRYPTION_TYPE:
        logging.error(f"Expected '{ENCRYPTION_TYPE}' encryption, got: {encryption}")
        raise Exception("Unexpected encryption type")

    id += 1
    m = hashlib.md5()
    h1 = f"{user}: "**********":{password}"
    m.update(h1.encode('UTF-8'))
    h1md5 = m.hexdigest().upper()

    m = hashlib.md5()
    h2 = f"{user}:{random}:{h1md5}"
    m.update(h2.encode('UTF-8'))
    h2md5 = m.hexdigest().upper()

    second = {
        "method": LOGIN_METHOD,
        "params": "**********": user, "password": h2md5, "clientType": CLIENT_TYPE, "loginType": "Direct", "authorityType": AUTHORITY_TYPE},
        "id": id,
        "session": session,
    }
    resp = make_request(loginURL, second)
    if not resp['result']:
        logging.error("Login failed")
        raise Exception("Login failed")

    return session

def control_light(session, io_type):
    global id
    id += 1
    control = {
        "method": CONTROL_METHOD,
        "params": {"channel": 0, "info": [{"Type": 1, "IO": io_type, "TriggerMode": 2}]},
        "id": id,
        "session": session
    }
    make_request(rpcURL, control)

def logout(session):
    global id
    id += 1
    logout = {"method": LOGOUT_METHOD, "params": None, "id": id, "session": session}
    make_request(rpcURL, logout)
    logging.info("Logged out")

def main():
    try:
        session = login()
        logging.info(f"Logged in with session: {session}")
        control_light(session, IO_TYPE_ON)  # Turn light on
        logging.info(f"Camera light is on")
        time.sleep(3)
        control_light(session, IO_TYPE_OFF)  # Turn light off
        logging.info(f"Camera light is off")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logout(session)

if __name__ == "__main__":
    main()
)
