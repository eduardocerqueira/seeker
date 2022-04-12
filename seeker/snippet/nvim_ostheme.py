#date: 2022-04-12T17:15:10Z
#url: https://api.github.com/gists/65a5505aae5afb08db5880b7c045cacd
#owner: https://api.github.com/users/dakyskye

#!/usr/bin/python3

import subprocess
import os
import re

if __name__ == '__main__':
    cmd = 'defaults read -g AppleInterfaceStyle'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    theme = 'dark' if bool(proc.communicate()[0]) else 'light'

    with open(f"{os.environ['HOME']}/.config/nvim/plugins.vim", "r+") as f:
        content = f.read()
        content = re.sub('set background=(light|dark)', f'set background={theme}', content)
        f.seek(0)
        f.truncate()
        i = f.write(content)
        if i != len(content):
            print("it seems we didn't manage to fully write the vim config...")
