#date: 2024-07-11T17:01:47Z
#url: https://api.github.com/gists/b992a51ca8966e27610498985ddb9fe8
#owner: https://api.github.com/users/openbrian

import sys
from random import random

# [electoral votes, chance biden, chance trump
state = {
  "wi": [10, 46, 48],
  "mi": [15, 45, 47],
  "pa": [19, 44, 48],
  "nv": [ 6, 43, 49],
  "az": [11, 43, 48],
  "ga": [16, 43, 49],
  "nc": [16, 43, 48],
}


biden_base = 226
trump_base = 219


def vote():
  bi = biden_base
  tr = trump_base
  for st in state:
    ev = state[st][0]
    total = state[st][1] + state[st][2]
#    total = 100
    biden_chance = state[st][1] / total
    trump_chance = (state[st][2] / total)
    trump_chance = biden_chance + trump_chance
    r = random()
    if r < biden_chance:
      bi += ev
    elif r < trump_chance:
      tr += ev
  if bi > 270:
    return 'b'
  if tr > 270:
    return 't'
  return 'o'



pres = {
  "b": 0,
  "t": 0,
  "o": 0,
}

n = int(sys.argv[1])
for i in range(n):
  pres[vote()] += 1

print()
print()
print(pres)

for p in pres:
  print(f"{p} {pres[p]/n}")