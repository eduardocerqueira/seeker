#date: 2025-04-18T16:48:06Z
#url: https://api.github.com/gists/99c28617cc787e77d1481aaec8cacbc9
#owner: https://api.github.com/users/rec

funcs = []

for i in range(10):
  def func(i=i):
    return i
  funcs.append(func)
  
print(*(f() for f in funcs))
