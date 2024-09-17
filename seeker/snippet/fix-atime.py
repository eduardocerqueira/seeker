#date: 2024-09-17T17:09:25Z
#url: https://api.github.com/gists/10ab9650a38e8556c74a1dd4876cd60c
#owner: https://api.github.com/users/meeb

#!/usr/bin/python3

import os
import time
from pathlib import Path

this_dir = os.path.dirname(os.path.realpath(__file__))
silly_time = time.mktime(time.strptime('2038-01-01', '%Y-%m-%d'))

for root, dirs, files in os.walk(this_dir):
    rootpath = Path(root)
    for f in files:
        filepath = rootpath / f
        if not os.path.isfile(filepath):
            continue
        access_time = os.path.getatime(filepath)
        # if the file access time is > 2038-01-01 then touch it
        if access_time > silly_time:
            print(f'Fixing future access time file: {filepath} ({access_time} > {silly_time})')
            filepath.touch()