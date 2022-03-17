#date: 2022-03-17T16:59:06Z
#url: https://api.github.com/gists/e7b1427433aa0eb681ceb9c19912d55c
#owner: https://api.github.com/users/hnguyen094

#!/usr/bin/python
## This script assumes that both npmrc and upmconfig.toml are already there, and the only change needed
## is to refresh an old npm token.

## How to use:
## 1) go to NPM website & sign in
## 2) copy & paste the auth token command into terminal ("npm config set..._authToken...")
## 3) run this code (python update_upm_token.py)

import sys
import re
from pathlib import Path

NPMRC=".npmrc"
UPMTOML=".upmconfig.toml"

token = ""
with open(Path.home().joinpath(NPMRC), "r") as f:
    for line in f.readlines():
        if "_authToken" in line:
            token= line[line.index("=")+1:].strip()
            break
print("token from " + NPMRC + ": " + token)

data = []
with open(Path.home().joinpath(UPMTOML), "r") as f:
    data = f.readlines()
    for i, line in enumerate(data):
        if "=" in line:
            data[i] = line[:line.index("\"")] + "\"" + token + "\""
            break

with open(Path.home().joinpath(UPMTOML), "w") as f:
    f.writelines(data)

print("updated " + UPMTOML)
