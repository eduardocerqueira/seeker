#date: 2023-08-15T16:59:54Z
#url: https://api.github.com/gists/3b9448f2ee81c02646695792d6129439
#owner: https://api.github.com/users/lvijay

# gist for the YouTube video https://youtu.be/SrYHB0YX5_U

import hashlib
import zlib

TEMPLATE = 'tree 62ee7f49375b274e064e4277b10f044fca62f144\nparent e4fe1d2e9d8e123586da04c86cf74983ae399385\nauthor Vijay Lakshminarayanan <laksvij@hawk.iit.edu> 1597940579 +0530\ncommitter Vijay Lakshminarayanan <laksvij@hawk.iit.edu> 1597940579 +0530\n\n'

def calc(msg):
  commit = TEMPLATE + msg + "\n"
  txt = (f'commit {len(commit)}\x00{commit}')
  ascbytes = txt.encode('us-ascii')
  return hashlib.sha1(ascbytes).hexdigest()
    
for i in range(300000, 400000):
  cmsg = f'commit #{i} contains c0ffee'
  if 'c0ffee' in (c := calc(cmsg)):
    print(f"{i=}")
    print(f"{cmsg=}")
    print(f"{c=}")
