#date: 2025-02-26T17:02:36Z
#url: https://api.github.com/gists/ee92a143692c991cf7a44c7bf4f8a9b6
#owner: https://api.github.com/users/bolu61

# Chapter 1: immediate evaluation with optional composition
@lambda f: not f()
def true() -> bool:
  return False
  
assert true == True # ???

# Chapter 2: single line composition
from functools import partial

@partial(partial, lambda f, *a, **k: not f(*a, **k))
def inv(x: bool) -> bool:
  return x

assert inv(true) = not true # ???

# Chapter 3: [something useful maybe](https://stackoverflow.com/a/24047214)
from functools import reduce

@partial(partial, reduce)
def compose(f, g):
  def composed(*a, **k):
    return f(g(*a, **k))
  return composed
