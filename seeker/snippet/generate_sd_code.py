#date: 2024-12-11T17:07:16Z
#url: https://api.github.com/gists/4245ec7888a61a1ad0ca49c5d4b04be0
#owner: https://api.github.com/users/rccarlson

"""
Generates today's code for accessing the Covenant Eyes developer menu.
Developer code is a truncated MD5 hash of the current date, with some formatting applied
"""

import hashlib, datetime

# get current date as ddmmyyyy
today = datetime.datetime.now().strftime(R"%d%m%Y")

# generate md5 hash of current date, surrounded by "SDsalt"
md5_hash = hashlib.md5()
md5_hash.update(f"SDsalt{today}SDsalt".encode('utf-8'))
hex_code = md5_hash.hexdigest()

# swap letters in the hex code with numbers
# truncate code to 9 characters
translation_table = str.maketrans('abcdef','012345')
code = hex_code[0:9].translate(translation_table)

# format the code in groups of 3, separated by dashes
formatted_code = f"{code[0:3]}-{code[3:6]}-{code[6:9]}"
print(formatted_code)