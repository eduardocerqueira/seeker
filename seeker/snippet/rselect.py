#date: 2024-02-19T17:02:56Z
#url: https://api.github.com/gists/98fb0f67e3df72dde5d1ddc6b1f59f3a
#owner: https://api.github.com/users/sauvikgon

import random

def partition(arr, low, high):
  pivot_i = random.randint(low, high)
  arr[low], arr[pivot_i] = arr[pivot_i], arr[low]
  pivot = arr[low]
  i = low + 1
  for j in range(low + 1, high + 1):
    if arr[j] < pivot:
      arr[i], arr[j] = arr[j], arr[i]
      i+=1
  arr[low], arr[i-1] = arr[i-1], arr[low]
  return i-1
  
def rselect(arr, low, high, i):
  if high == low:
    return arr[low]
  pivot_i = partition(arr, low, high)
  if i == pivot_i:
    return arr[pivot_i]
  elif i < pivot_i:
    return rselect(arr, low, pivot_i-1, i)
  else:
    return rselect(arr, pivot_i+1, high, i)

def  select(arr, i):
  return rselect(arr, 0, len(arr)-1, i)

arr = [82,22,26,12,41,62,17,28,91,10]
i=5
print(f"The {i}th smallest element is {select(arr, i)}")