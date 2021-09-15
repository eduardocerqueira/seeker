#date: 2021-09-15T17:02:56Z
#url: https://api.github.com/gists/184e560d8b2a872eb2cfd286f6133ae2
#owner: https://api.github.com/users/TruthyCode

import base64
from pathlib import Path

while True:
    file = Path(input("Enter the Filename: "))
    try:
        validation = file.is_file()
        if validation:
            print("Validated...\n")
            break
        else:
            print("Enter a valid FileName In your current directory...\n")

    except Exception as ex:
        print("An Exception Occurred")

with open(file) as text:
    b64str = b""
    for line in text:
        b64line = base64.b64encode(bytes(line, "ASCII"))
        b64str += b64line
print(b64str)
