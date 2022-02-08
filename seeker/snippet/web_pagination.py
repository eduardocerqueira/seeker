#date: 2022-02-08T17:03:08Z
#url: https://api.github.com/gists/645e87ba976a0ae059b819f9751c4bed
#owner: https://api.github.com/users/fmorenovr

def fetchItemsToDisplay(items, sortParameter, sortOrder, itemsPerPage, pageNumber):
  # Write your code here
  # reverse=True -> descending
  sorted_items = sorted(items, key = lambda i: i[sortParameter] if sortParameter==0 else int(i[sortParameter]) , reverse=sortOrder)

  print("items:", items)
  print("parameters:", sortParameter, sortOrder)
  print("sorted:",sorted_items)

  result = []

  for item_ in sorted_items[(pageNumber)*itemsPerPage:(pageNumber+1)*itemsPerPage]:
    result.append(item_[0])

  return result