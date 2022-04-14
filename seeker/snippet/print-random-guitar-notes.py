#date: 2022-04-14T16:47:51Z
#url: https://api.github.com/gists/aac4aab4ca4be758f8a6de9709d1dd61
#owner: https://api.github.com/users/vsinha

import random
import time

sleep_sec = 1
natural_notes = ["A", "B", "C", "D", "E", "F", "G"]
prev = None


def print_randomly():
    global prev

    curr = random.choice(natural_notes)
    if prev == curr:
        curr = random.choice(natural_notes)

    prev = curr
    print(curr)


while True:
    print_randomly()
    time.sleep(sleep_sec)