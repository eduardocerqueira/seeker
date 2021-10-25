#date: 2021-10-25T17:05:06Z
#url: https://api.github.com/gists/f1bd5184dc68760b31125c2e8e1efa5d
#owner: https://api.github.com/users/OmerCelikel

import math
result = 1
#math.factorial(int(num))
for i in range(30):
  result = math.factorial(365)/((math.factorial(365-i)*(365**i)))
  if result < 0.5:
    print(result)
    print(i)
    break