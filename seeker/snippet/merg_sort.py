#date: 2025-04-03T16:54:24Z
#url: https://api.github.com/gists/583f7bbabf368ae1cc90cefb49615a3d
#owner: https://api.github.com/users/alsidneio

def merge_sort(nums):
    if len(nums) < 2: 
        return nums
    midpoint = len(nums)//2
    return merge(merge_sort(nums[:midpoint]),merge_sort(nums[midpoint:]))

def merge(first, second):
  final = []
  i,j = 0,0
  while i < len(first) and j < len(second): 
      if first[i] <= second[j]: 
          final.append(first[i])
          i+=1
      else: 
          final.append(second[j])
          j+=1

  if i< len(first): 
      final.extend(first[i:])
  if j< len(second): 
      final.extend(second[j:])
  return final