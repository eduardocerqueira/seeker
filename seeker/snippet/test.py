#date: 2022-05-02T17:13:12Z
#url: https://api.github.com/gists/7644b2395e3c8ed420f049c807d6732a
#owner: https://api.github.com/users/tsagaanbar

import time

def func_1():
    [ print('', end='') for i in range(1000000) ]


def func_2():
    for i in range(1000000):
        print('', end='')


def test_for_n_times(n, foo):
    for i in range(10):
        st = time.time()
        foo()
        print(time.time() - st)


test_for_n_times(10, func_1)

print('-'*20)

test_for_n_times(10, func_2)
