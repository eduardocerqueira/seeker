#date: 2021-09-13T17:04:40Z
#url: https://api.github.com/gists/72b82ef2a107137d6f31287d439e50d2
#owner: https://api.github.com/users/gustavomotadev

import re

def check_duplicate(str_in):
    mat = re.match(check_duplicate.pat, str_in)
    if mat:
        return mat.group(1)
    else:
        return str_in
check_duplicate.pat = re.compile(r'([a-zA-Z0-9]+[\w\s-]*).*\1')

print(check_duplicate('Samsung Galaxy S20 - Green - Samsung Galaxy S20 - Green'))