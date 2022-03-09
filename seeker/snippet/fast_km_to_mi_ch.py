#date: 2022-03-09T16:54:06Z
#url: https://api.github.com/gists/26ed9436bfb317847139a6cc2cb007dd
#owner: https://api.github.com/users/tbruckbauer

from math import floor

while True:
    print("Enter the amount of km to convert to mi and ch")
    km = float(input())
    mi = km * 0.62137
    ch = (mi % 1) * 80
    print(str(km) + "km equals " + str(floor(mi)) + "mi, " + str(round(ch)) + "ch.\nAbort with ctrl+d")