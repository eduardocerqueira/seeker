#date: 2023-01-17T16:52:22Z
#url: https://api.github.com/gists/a00d8ac2321e6b8bf450a8c1fb95b31d
#owner: https://api.github.com/users/ygtadk

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Gigachad Sıralama Algoritması a.k.a. BogoSort

import random
import time
start_time = time.time()


def is_sorted(a):
    for i in range(len(a) - 1):
        if a[i] > a[i+1]:
            return False
    return True


def gigachad_sort(a, x):
    while not is_sorted(a):
        random.shuffle(a)
        x += 1
        print("Yineleme: ", x)
    return a


if __name__ == '__main__':
    a = [random.randint(0, 1000) for i in range(6)]
    print("--> Dizi: ", gigachad_sort(a, 0))

print("--> %s saniye sürdü. <--" % (time.time() - start_time))
