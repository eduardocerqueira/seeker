#date: 2023-11-03T16:51:46Z
#url: https://api.github.com/gists/89c35b8bd5cea9512d4e606ecaa56340
#owner: https://api.github.com/users/lostquix

from time import sleep

import pyautogui as pg

# This code is used to open URL in firefox
# browser

import webbrowser

# To take the URL as input from the user.

link = 'https://web.whatsapp.com/'

# Passing firefox executable path to the
# Mozilla class.
firefox = webbrowser.Mozilla("C:\\Program Files\
\Mozilla Firefox\\firefox.exe")

# Using open() function to display the URL.
firefox.open(link)

sleep(11)
pg.click(x=297, y=265)

sleep(0.50)
pg.write("Contact name")

sleep(1)
pg.click(x=319, y=443)

# Loop
for message in range(0, 100):

    # message
    pg.write("Hello")

    sleep(0.50)
    pg.press("Enter")
    