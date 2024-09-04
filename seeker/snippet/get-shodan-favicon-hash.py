#date: 2024-09-04T17:00:18Z
#url: https://api.github.com/gists/f8d4f1dfab26e3b3acdc46eadf99cfac
#owner: https://api.github.com/users/imsroot

# https://twitter.com/brsn76945860/status/1171233054951501824
pip install mmh3

-----------------------------
# python 2
import mmh3
import requests
 
response = requests.get('https://cybersecurity.wtf/favicon.ico')
favicon = response.content.encode('base64')
hash = mmh3.hash(favicon)
print hash

-----------------------------

# python 3

import mmh3
import requests
import codecs
 
response = requests.get('https://cybersecurity.wtf/favicon.ico')
favicon = codecs.encode(response.content,"base64")
hash = mmh3.hash(favicon)
print(hash)