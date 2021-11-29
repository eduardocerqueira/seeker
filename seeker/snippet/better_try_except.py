#date: 2021-11-29T17:01:37Z
#url: https://api.github.com/gists/082662b48e4c1de3abae8d12e63d53bf
#owner: https://api.github.com/users/garland3

import traceback
try:
  raise Exception("my error")
except Exception as e:
  print(e)
  print(traceback.format_exc())