#date: 2025-07-10T17:16:48Z
#url: https://api.github.com/gists/7c39cbdd5cd1d55212d59457583d56bb
#owner: https://api.github.com/users/mypy-play

from typing import *
import os

r: os.stat_result = os.stat('.')
print(r.st_ctime)