#date: 2023-07-17T17:01:47Z
#url: https://api.github.com/gists/15d124a63030c5743c50895926d1e221
#owner: https://api.github.com/users/spawnmarvel

import urllib.parse
# abcde/fghi+jklmn=opq
# abcde%2Ffghi%2Bjklmn%3Dopq
inp = "abcde/fghi+jklmn=opq"
print(inp)
rv = urllib.parse.quote(inp, safe='')
print(rv)