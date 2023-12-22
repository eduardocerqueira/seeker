#date: 2023-12-22T16:57:31Z
#url: https://api.github.com/gists/47940ab354425fc2df5ff9ae347d269c
#owner: https://api.github.com/users/amalrajan

import sys

try:
    sys.stdin = open(sys.path[0] + "/input.txt", "r")
    sys.stdout = open(sys.path[0] + "/output.txt", "w")
except FileNotFoundError:
    pass