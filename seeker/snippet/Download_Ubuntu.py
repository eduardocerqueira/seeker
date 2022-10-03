#date: 2022-10-03T17:25:45Z
#url: https://api.github.com/gists/3a4312c68a6ad4ad06d840ba07498376
#owner: https://api.github.com/users/1RaY-1

# Simple Python program to download latest Ubuntu LTS .iso file from official website (for amd64).

# [!] This script supports python version '3.10' or higher

import requests
import webbrowser
from time import sleep

try:
    from bs4 import BeautifulSoup as BS 
except ImportError:
    exit("Need 'bs4' module to be installed.")

try :
    r = requests.get("https://ubuntu.com/download/desktop")
    soup = BS(r.text, 'html.parser')
    v =  soup.find_all('h2')[0].get_text() 
except :
    exit("Check internet connection, please!")

def print_version():
    # here's stored the LTS latest version
    print("Latest Ubuntu LTS version is:\n" + str(v))

def make_clear_version():
    global v
    v = v.replace("Ubuntu", "")
    v = v.replace("LTS", "")
    v = v.replace(" ", "")

def ask_if_install():
    choice = input("\nDo you wanna download it?\n")

    match choice:
        case "y" | "yes" | "Y" | "YES": # ' | ' is like " or "

            print("Opening download link in browser...")
            sleep(0.8)
            new_url = f"https://ubuntu.com/download/desktop/thank-you?version={v}&architecture=amd64"
            webbrowser.open(new_url)
        case _:
            exit()

if __name__ == "__main__":
    print_version()
    make_clear_version()
    ask_if_install()
