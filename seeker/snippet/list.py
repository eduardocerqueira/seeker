#date: 2024-04-04T16:56:31Z
#url: https://api.github.com/gists/f8c130aa9579969806859ecef2988bab
#owner: https://api.github.com/users/preston-56

# ([1,2,3,4,5,6,7],3) === [[1,2,3],[3,4,5],[6,7]]
# ([1,2,3,5,6,7],1) == [[1],[2],[3],[5],[6],[7]]
def split_list(lst,n):
  result = []
  for i in range(0, len(lst),n):
    result.append(lst[i:i+n])
  return result

# Test cases
print(split_list([1,2,3,4,5,6,7],3))
print(split_list([1,2,3,5,6,7],1))