#date: 2022-04-06T17:13:33Z
#url: https://api.github.com/gists/14173383328d5048a95f96ed90a11ff6
#owner: https://api.github.com/users/sertdfyguhi

import re

def generate(name: str):
    return (name[0] + re.sub(
        '[aeiou ]',
        '',
        name[1:-1],
        flags=re.I
    ) + name[-1]).upper()

print(generate(input('name: ')))