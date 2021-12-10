#date: 2021-12-10T17:18:36Z
#url: https://api.github.com/gists/2d24e3c303361c04a35687f59d01cb44
#owner: https://api.github.com/users/jac18281828

import tokenize
from io import StringIO


with open('filename', 'r') as filestream:
  for line in filestream.readlines():
    for token in tokenize.generate_tokens(StringIO(line).readline):
      print(token)