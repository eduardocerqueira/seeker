#date: 2022-01-11T17:21:17Z
#url: https://api.github.com/gists/b9570fe21b0315c6288ff7432f5a4b73
#owner: https://api.github.com/users/dilettagoglia

# MERGESORT ALGORITHM

import numpy

x=(numpy.random.randint(0, 33))
y=(numpy.random.randint(0, 33))
A=numpy.random.randint(x, size=y)
temp = numpy.zeros(len(A), dtype=int)

def merge(V, l, c, r):
  i = k = l
  j = c+1
  while i<=c and j<=r:
    if V[i] <= V[j]:
      temp[k] = V[i]
      i = i+1
    else:
      temp[k] = V[j]
      j= j+1
    k = k+1
  while i<=c:
    temp[k] = V[i]
    i = i+1
    k = k+1
  while j<=r:
    temp[k] = V[j]
    j = j+1
    k = k+1
  for k in range(l, r+1):
    V[k] = temp[k]

def mergesort(V, l, r):
  if l>=r:
    return
  m = (l+r)//2
  mergesort(V, l, m)
  mergesort(V, m+1, r)
  merge(V, l, m, r)
  return V