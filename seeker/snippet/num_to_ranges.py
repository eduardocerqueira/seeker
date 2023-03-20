#date: 2023-03-20T17:07:40Z
#url: https://api.github.com/gists/6398b648fc2b18bbdf2ee4a892917feb
#owner: https://api.github.com/users/jcha-ultra

# Define a function that takes a set of integers as input
def num_to_ranges(num_set):
  # Initialize an empty list to store the ranges
  ranges = []
  # Sort the input set in ascending order using a built-in function
  sorted_set = sorted(num_set)
  # Initialize variables to store the start and end of each range
  start = None
  end = None
  # Loop through the sorted set
  for num in sorted_set:
    # If start is None, it means we are at the first element or a new range has started
    if start is None:
      # Set both start and end to the current element
      start = num
      end = num
    # Else, check if the current element is consecutive to the previous one (end)
    elif num == end + 1:
      # If yes, update the end to the current element
      end = num
    # Else, it means we have reached the end of a range
    else:
      # Append the range as a tuple to the list
      ranges.append((start, end))
      # Reset both start and end to the current element for a new range
      start = num 
      end = num 
  # After looping through all elements, append the last range to the list if it exists  
  if start is not None:
    ranges.append((start, end))
  # Return the list of ranges as output  
  return ranges

# Test cases

print(num_to_ranges({4,-3 ,6 ,2 ,71 ,3 ,70 ,72 ,1000 ,5})) 
# Output: [(-3,-3), (2,6), (70,72), (1000,1000)]

print(num_to_ranges({1}))
# Output: [(1,1)]

print(num_to_ranges({10,-5}))
# Output: [(-5,-5), (10 ,10)]

print(num_to_ranges({9 ,8 ,7 ,6 ,5 ,4}))
# Output: [(4 ,9)]