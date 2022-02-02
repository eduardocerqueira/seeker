#date: 2022-02-02T17:03:15Z
#url: https://api.github.com/gists/7a7241b26a5c62838c551b03d172e84f
#owner: https://api.github.com/users/erbenpeter

from random import randint


for t in range(30):
  N = randint(10, 100)
  M = randint(30, 150)
  l1 = [randint(1, 1000) for _ in range(N)]
  l2 = [randint(100, 2000) for _ in range(M)]
  print(' '.join([str(x) for x in sorted(l1)]))
  print(' '.join([str(x) for x in sorted(l2)]))
