#date: 2023-06-07T16:45:10Z
#url: https://api.github.com/gists/3ff027aea10566726ec5a615fab5fddd
#owner: https://api.github.com/users/prehensilecode

#!/usr/bin/env python3
import sys
import os
import re
import pwd

# some of the User_Aliases lines contain buggy usernames 
# which have "a" at the end of the username

broken_usernames = set()
usernames = set()
useralias_pat = re.compile(r'^User_Alias')
with open('sudoers', 'r') as sudoersfile:
    for l in sudoersfile:
        if useralias_pat.match(l):
            usernames.update(l.strip().split('=')[-1].strip().split(','))

for u in usernames:
    try:
        pwd.getpwnam(u)
    except Exception as e:
        broken_usernames.add(u)

print('Broken usernames:')
for u in broken_usernames:
    print('\t', u)