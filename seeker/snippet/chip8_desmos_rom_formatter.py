#date: 2024-04-01T16:58:32Z
#url: https://api.github.com/gists/d92d644174bc1a8b250df19b7432b7ff
#owner: https://api.github.com/users/typecasto

import sys

with open(sys.argv[1], "rb") as file:
  print("\\left[", end="")
  print(*file.read(), sep=",", end="")
  print("\\right]", end="")