#date: 2022-07-27T17:20:56Z
#url: https://api.github.com/gists/019d6ee8ab652567b5059b456dda67dc
#owner: https://api.github.com/users/joachimesque

#!/usr/bin/env python3

"""Downloads and checks external front-end libs

Avoid CDNs for front-end libraries:
https://blog.wesleyac.com/posts/why-not-javascript-cdn
"""

import os
import base64
import hashlib

import requests

# https://www.srihash.org/
libs = [
    (
        "https://unpkg.com/chota@0.8.0/dist/chota.css",
        "chota-0.8.0.css",
        "sha384-rn488xVSy52er61VbV56rSIPTxXtCTcectcsH/0VOC9RwoajWF3O4ukT8bmZVCNy",
    ),
    (
        "https://unpkg.com/htmx.org@1.8.0/dist/htmx.js",
        "htmx-1.8.0.js",
        "sha384-mrsv860ohrJ5KkqRxwXXj6OIT6sONUxOd+1kvbqW351hQd7JlfFnM0tLetA76GU0",
    ),
]

EXPORT_DIR = "../vendor/"

for lib in libs:
    destination = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), EXPORT_DIR, lib[1]
    )
    resource_data = requests.get(lib[0]).content

    if lib[2] != "":
        integrity_checksum = base64.b64encode(
            hashlib.sha384(resource_data).digest()
        ).decode("utf-8")
        if integrity_checksum == lib[2].rsplit("-", maxsplit=1)[-1]:
            print(f"{lib[1]} SRI check OK")
        else:
            raise Exception(f"SRI check failed for {lib[1]}")
    else:
        print(f"Warning: Could not check SRI for {lib[1]}")

    with open(destination, "wb") as file:
        file.write(resource_data)
