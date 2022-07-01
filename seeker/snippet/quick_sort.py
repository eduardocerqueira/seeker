#date: 2022-07-01T17:06:17Z
#url: https://api.github.com/gists/4dc020cf7daa38a0182c8f3fcc0a9875
#owner: https://api.github.com/users/adrolc

'''
Time efficiency test

Algorithm: quicksort. Minimum execution time: 0.10355029999999998
Algorithm: quicksort2. Minimum execution time: 0.17530590000000013
'''

import random

# not in-place
def quicksort(array):
    if len(array) < 2:
        return array

    low, same, high = [], [], []
    pivot = array[random.randint(0, len(array) - 1)]

    for item in array:
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)
    return quicksort(low) + same + quicksort(high)


# in-place
def quicksort2(array, low=0, high=None):
    if high == None:
        high = len(array) - 1

    if low >= high:
        return

    i, j = low, high
    pivot = array[random.randint(low, high)]

    while i <= j:
        while array[i] < pivot:
            i += 1
        while array[j] > pivot:
            j -= 1

        if i <= j:
            array[i], array[j] = array[j], array[i]
            i, j = i + 1, j - 1

    quicksort2(array, low, j)
    quicksort2(array, i, high)
