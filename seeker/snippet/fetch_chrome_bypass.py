#date: 2024-07-12T17:08:46Z
#url: https://api.github.com/gists/321ac7189d0a8f30057eea5a8241940b
#owner: https://api.github.com/users/tdoylend

# Kudos to the Chromium devs for needlessly obfuscating how to access a site with an invalid HTTPS cert. /s

import requests
import re
from base64 import b64decode

data = requests.get('https://chromium.googlesource.com/chromium/src/+/refs/heads/main/components/security_interstitials/core/browser/resources/interstitial_large.js?format=TEXT').content

data = b64decode(data)
#print(data)
#open('t','wb').write(data)

r = re.search(r"BYPASS_SEQUENCE.*?'(.*?)'".encode('ascii'), data)

print('Current Chromium bypass: '+b64decode(r.group(1)).decode('ascii'))
