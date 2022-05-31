#date: 2022-05-31T17:02:54Z
#url: https://api.github.com/gists/1e0c1dcc110c2a74aaa1887dede8904e
#owner: https://api.github.com/users/jasalt

#!/usr/bin/env python

import glob
import os
import sys

def print_help():
    print(''' Replaces file shortcuts in given path with files they represent (on MacOS).
    Directory shortcuts are skipped. Requires Python 3.10.

    Usage:

    python deshortcut_path.py   # process current workdir
    python deshortcut_path.py   # process given dir
    ''')

match length := len(sys.argv):
    case 1:
        print("No argument supplied, using current workdir.")
        search_path = os.getcwd()
    case 2:
        search_path_relative = sys.argv[1]
        if not os.path.isdir(search_path_relative):
            print("ERR: Argument is not a directory.")
            print_help()
            exit(1)
        search_path = os.path.abspath(search_path_relative)
    case _:
        print("ERR: Multiple arguments not supported.")
        print_help()
        exit(1)

print(f"Processing path {search_path}")

filepaths = os.listdir(search_path)

from mac_alias import resolve_osx_alias, isAlias
from shutil import copyfile

count_successful = 0
for file in filepaths:
    try:
        if not isAlias(file):
            continue
    except Exception as e:
        # TODO throws exception for some reason (MacOS Monterey), isAlias is buggy...
        print(f"Skipped something, maybe not alias {file}")
        continue

    file_alias = file
    file_original = resolve_osx_alias(file_alias)
    
    if not os.path.isfile(file_original):
        print(f"Skipped shortcut with missing original file {file_original}")
        continue

    os.remove(file_alias)  # remove alias/shortcut file
    copyfile(file_original, file_alias)  # move original file in it's place
    print(f"Replaced alias {file_alias} with {file_original}")
    count_successful += 1

print(f'Replaced {count_successful} shortcuts with original files.')    
