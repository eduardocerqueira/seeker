#date: 2023-03-10T17:01:49Z
#url: https://api.github.com/gists/1bcd6465268651c38dada237ae60d577
#owner: https://api.github.com/users/growingpenguin

import random
def is_sorted(a):
    n = len(a)
    for i in range(0, n-1):
        if (a[i] > a[i+1]):
            return False
    return True

def shuffle(a):
    n = len(a)
    for i in range(0, n):
        r = random.randint(0, n-1)
        a[i], a[r] = a[r], a[i]

def permutation_sort(a):
    n = len(a)
    while (is_sorted(a) == False):
        shuffle(a)
    return a

ds=[3,1,2]
print(permutation_sort(ds))
