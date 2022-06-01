#date: 2022-06-01T17:02:34Z
#url: https://api.github.com/gists/8fefcfa4d9e374357a662bb78e547a5b
#owner: https://api.github.com/users/hisashi-ito

# -* coding:utf-8 -*-
import loop as l
import time

def loop():
    for i in range(50000):
        for j in range(50000):
            sum = i + j
    return sum


# python
time_sta = time.time()
loop()
print("python elapsed time: {}".format(time.time() - time_sta))

# cython
time_sta = time.time()
l.loop()
print("cythonn elapsed time: {}".format(time.time() - time_sta))
