#date: 2022-04-04T17:04:50Z
#url: https://api.github.com/gists/cf08ef57c473f0f206931781fe2b9668
#owner: https://api.github.com/users/tashpjak

a = [ 1, 3, 6, 8,  5, 6, 4, 8]  # [0, 1, 3, 5, 6, 7, 8]


def merge(array):
    if len(array) <= 1:
        return array
    left = array[0:len(array) // 2]
    right = array[len(array) // 2:]

    merged_left = merge(left)
    merged_right = merge(right)
    return sort(merged_left, merged_right)


def sort(left, right):
    result = []
    n = len(left) + len(right)
    while len(result) != n:
        if len(right) == 0:
            result += left
            break
        if len(left) == 0:
            result += right
            break
        if left[0] >= right[0]:
            result.append(right[0])
            right.pop(0)
        else:
            result.append(left[0])
            left.pop(0)

    return result


print(merge(a))










fibonachi = [1,1]

# def fib(array):
#     array.append(array[-2] + array[-1])
#
#
# for i in range(10):
#     fib(fibonachi)



import sys

sys.setrecursionlimit(1000000000)

def rec_fib(array, number):
    array.append(array[-2] + array[-1])
    if number > 0:
        return rec_fib(array, number - 1)
    return array


rec_fib(fibonachi, 1000)
print(fibonachi)












a = [1, 5, 67, 87, 8]


class Node:
    def __init__(self, value, left, right):
        self.value = value
        self.right = right
        self.left = left
        print("created")


a1 = Node(5, None, None)


import numpy as np

num = [[519432, 525806],
       [632382, 518061],
       [78864, 613712],
       [466580, 530130],
       [780495, 510032]]
maxi = 0
for i in range(5):
    res = np.log(num[i][0]) * num[i][1]
    if res > maxi:
        maxi = res
        index = i
        ind = i

print(ind)
