#date: 2022-10-19T17:19:52Z
#url: https://api.github.com/gists/9f7e735b8fbc54badc0007b95541528b
#owner: https://api.github.com/users/yennster

import yaml
import sys
import os
import re
from collections import defaultdict
from os import path, linesep

exit_code = int(0)   # 0 = success
redirects_errors = ""
summary_errors = ""

INLINE_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
with open('SUMMARY.md', 'r') as file:
    summary_file = file.read().replace('\n', '')
summary_links = list(INLINE_LINK_RE.findall(summary_file))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

with open(".gitbook.yaml", "r") as stream:
    exit_code = 0
    redirects_errors = ""
    summary_errors = ""
    data = yaml.safe_load(stream)

"""
Check if all redirects paths used in .gitbook.yaml exist within the repo.
"""
print(bcolors.OKCYAN + "\U0001F4AD Check .gitbook.yaml redirect paths files existence...")
for key, value in data['redirects'].items():
    try:
        if not path.exists(value): 
            #print(bcolors.OKGREEN + "OK\t" + value)
            raise ValueError
    except ValueError:
        #print(bcolors.FAIL + "ERR\t" + value)
        exit_code = int(2)
        redirects_errors = redirects_errors + os.linesep + value
        pass

"""
Now check whether all redirects paths used in .gitbook.yaml are present in SUMMARY.md.
This check is necessary because if the paths are not referenced anywhere in SUMMARY.md,
then GitBook will not know where to redirect pages that no longer exist in the docs,
but are still cached in the Google search results, for example.
"""
print(bcolors.OKCYAN + "\U0001F4AD Check all redirect paths are referenced in SUMMARY.md...")
for key, value in data['redirects'].items():
    try:
        linted_value = value.replace('_', '\\_')
        res = [linted_value in list for list in summary_links]
        if not any(res):
            #print(bcolors.OKGREEN + "OK\t" + value)
            raise ValueError
    except ValueError:
        #print(bcolors.FAIL + "ERR\t" + value)
        exit_code = int(2)
        summary_errors = summary_errors + os.linesep + value
        pass

if exit_code is not int(0):
    print(bcolors.HEADER + "—————————————————————————————————————————————————————————————————")
    if redirects_errors:
        print(bcolors.WARNING + "\U0001F614 The following redirect paths do not exist in the " + bcolors.UNDERLINE + "repository" + bcolors.ENDC + bcolors.WARNING + ":")
        print(bcolors.FAIL + redirects_errors + os.linesep)
    if summary_errors:
        print(bcolors.WARNING + "\U0001F614 The following redirect paths do not exist in " + bcolors.UNDERLINE + "SUMMARY.md" + bcolors.ENDC + bcolors.WARNING + ":")
        print(bcolors.FAIL + summary_errors + os.linesep)
    print(bcolors.FAIL + "\U0000274C Checks completed with ERRORS.")
    sys.exit(exit_code)

print(bcolors.OKGREEN + "\U00002705 Checks completed SUCCESSFULLY.")
sys.exit(exit_code)