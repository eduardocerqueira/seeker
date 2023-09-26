#date: 2023-09-26T16:51:30Z
#url: https://api.github.com/gists/7f1618b705935064a8f90260f48b2d3e
#owner: https://api.github.com/users/Dolamu-TheDataGuy

from typing import Callable

def add(x: int, y: int) -> int:
  return a+b

def product(x: int, y: int) -> int:
  return a*b

def task(function: Callable[[int, int], int], a: int, b:int) -> int:
  return function(a, b)

answer_add = task(add, 3, 5)
answer_product = task(product, 3, 7)

print(answer_add, answer_product)