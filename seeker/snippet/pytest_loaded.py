#date: 2022-07-07T17:16:55Z
#url: https://api.github.com/gists/995aef117b41ae2adc10e24e24b60aa3
#owner: https://api.github.com/users/neocliff

from sys import modules as sys_modules

if 'pytest' in sys_modules:
    print("code running in PyTest!")
