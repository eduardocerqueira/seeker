#date: 2025-10-31T17:03:26Z
#url: https://api.github.com/gists/6b0fb11f5f409efbddd3f76703b582e6
#owner: https://api.github.com/users/idolpx

#!/usr/local/bin/python3
import os

list = []

dirlist = [os.getcwd() + '/']

# Build recursive list for dirs and files
while len(dirlist) > 0:
    for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
        dirlist.extend(dirnames)
        list.extend( map(lambda n: os.path.join(*n), zip([dirpath] * len(dirnames), dirnames)) )
        list.extend( map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)) )

# Iterate through list backwards and rename files and folders
for x in range(len(list) -1, -1, -1):
    item = list[x]
    print(f'{item} >>> {item.lower()}')
    os.rename(item, item.lower())