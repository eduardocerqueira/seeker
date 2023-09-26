#date: 2023-09-26T17:00:49Z
#url: https://api.github.com/gists/d522a57f3801e74b5f216f28b5d144c9
#owner: https://api.github.com/users/Dolamu-TheDataGuy

def multiplier(factor: int) -> int:
  def multiply(x: int):
    return x * factor
  return multiply

one = multiplier(3)
two = multiplier(5)

double = one(5)  # returns 15
tripple = two(2) # returns 10