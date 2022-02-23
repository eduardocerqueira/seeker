#date: 2022-02-23T17:13:05Z
#url: https://api.github.com/gists/4fa63bfedf0b3d9389550995505871c9
#owner: https://api.github.com/users/jowagner

#!/usr/bin/env python

import sys

offset = int(sys.argv[1])
img_name = sys.argv[2]

# skip current_pos
while True:
    line = sys.stdin.readline()
    if not line: break
    if line.startswith('#'): continue
    # reached first non-comment line
    # --> this is the current_pos line
    break

# process ranges
while True:
    line = sys.stdin.readline()
    if not line: break
    if line.startswith('#'): continue
    fields = line.split()
    if fields[2] == '+': continue
    assert fields[2] == '?'
    # found free area
    start = int(fields[0], 16)
    length = int(fields[1], 16)
    sys.stdout.write('fallocate -p -l %d -o %d %s\n' %(
        length, start + offset, img_name
    ))