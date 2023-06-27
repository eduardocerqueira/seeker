#date: 2023-06-27T16:51:40Z
#url: https://api.github.com/gists/bfbb3fef638426202474232fdca5b414
#owner: https://api.github.com/users/GuillermoIzquierdo

from random import randrange

def function():
  A = randrange(10)
  B = randrange(10)
  C = A * B
  while C != 4:
      print(A)
      print(C)
      A = randrange(10)
      B = randrange(10)
      C = A * B
  print("Success! A:{A_} B:{B_}".format(A_ = A, B_ = B))