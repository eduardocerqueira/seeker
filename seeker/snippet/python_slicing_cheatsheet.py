#date: 2025-07-11T17:05:06Z
#url: https://api.github.com/gists/9bcd1ca7660b4623145f86bd4115b207
#owner: https://api.github.com/users/GraceCindie

my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Get elements from index 2 up to (but not including) 7
print(my_list[2:7])    # Output: [2, 3, 4, 5, 6]

# Get elements from the start up to (but not including) 5
print(my_list[:5])     # Output: [0, 1, 2, 3, 4]

# Get elements from index 5 to the end
print(my_list[5:])     # Output: [5, 6, 7, 8, 9]

# Get a copy of the whole list
print(my_list[:])      # Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Get every other element
print(my_list[::2])    # Output: [0, 2, 4, 6, 8]

# Reverse the list
print(my_list[::-1])   # Output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]