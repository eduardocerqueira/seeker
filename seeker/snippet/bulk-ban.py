#date: 2022-08-30T16:43:35Z
#url: https://api.github.com/gists/e774e003de0593e5ab72ced49d0ea59f
#owner: https://api.github.com/users/nukeador

#!/usr/bin/env python3

"""
    Usage bulk-ban.py room-list.txt @userid:server.com
    
    You will need matrix-commander installed configured before
    https://github.com/8go/matrix-commander/
"""

import subprocess, sys

file_path = sys.argv[1]
user = sys.argv[2]

with open(file_path) as file:
    for line in file:
        room = line.replace('\n','')
        subprocess.run(['matrix-commander', '--room-ban', room, '--user', user])