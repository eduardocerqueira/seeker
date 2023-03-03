#date: 2023-03-03T17:01:32Z
#url: https://api.github.com/gists/4b54101380db8889c83937a2c1550516
#owner: https://api.github.com/users/AaronTraas

import random

# Precondition: list is a list of numbers
def is_ordered(list):
    last = None
    for el in list:
        if last is not None and el < last:
            return False
        last = el
    return True


# Precondition: list is a list of numbers, index1 and index2 are integers between 0 and len(list)
def swap_elements(list, index1, index2):
    list[index1], list[index2] = list[index2], list[index1]
    return list


# Precondition: list is a list of numbers
def stupid_sort(list):
    max = len(list)
    count = 0
    while not is_ordered(list):
        list = swap_elements(list, random.randrange(max), random.randrange(max))
        count += 1

    print("Total number of iterations:", count)
    return list


# Generate a list of 100 random integers between 0 and 1024
input_list = random.sample(range(0, 1024), 10)

print("Input list:", input_list)

sorted_list = stupid_sort(input_list)

print("Sorted list:", sorted_list)
