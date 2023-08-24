#date: 2023-08-24T16:31:53Z
#url: https://api.github.com/gists/8f691f9923348c33497c2f34c8d2f39a
#owner: https://api.github.com/users/asemic-horizon

from math import sqrt
def _hsmode(zs,f):

  if len(zs)==1: 
    return zs[0]
  elif len(zs)==2:
    return f(zs[0],zs[1])

  else:
    h = len(zs)//2
    if (zs[h]-zs[0]) <  (zs[-1]-zs[h+1]):
      return _hsmode(zs[:h],f)
    else:
      return _hsmode(zs[h:],f)

def calc_hsmode(xs,f):
    return _hsmode(sorted(xs),f)
