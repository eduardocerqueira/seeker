#date: 2021-11-19T16:55:37Z
#url: https://api.github.com/gists/82687c44d95c85258ef09b3363de4da8
#owner: https://api.github.com/users/chendonghp

from typing import List
import itertools


def bubble(list_):
    # in place
    for end in range(len(list_)-1,0,-1):
        for i in range(end):
            if list_[i]>list_[i+1]:
                list_[i], list_[i+1] = list_[i+1],list_[i]


# inplace
def selection(list_):
    if list_:
        for i in range(len(list_) - 1):
            min_index = i
            for j in range(i + 1, len(list_)):
                if list_[j] < list_[min_index]:
                    min_index = j
            list_[i], list_[min_index] = list_[min_index], list_[i]


def insertion(list_):
    if list_:
        for i in range(1, len(list_)):
            insert_val = list_[i]
            j = i - 1
            while j >= 0 and list_[j] > insert_val:
                list_[j + 1] = list_[j]
                j -= 1
            list_[j + 1] = insert_val


def shell(list_):
    def gapinsertion(list_, start, interval):
        # in-place sort
        for i in range(start, len(list_), interval):
            j = i
            insert_val = list_[j]
            while j - interval >= start and insert_val < list_[j - interval]:
                list_[j] = list_[j - interval]
                j -= interval
            list_[j] = insert_val

    interval = len(list_) // 2
    while interval > 0:
        for i in range(interval):
            gapinsertion(list_, i, interval)
        interval >>= 1


def merge(list_):
    len_ = len(list_)
    if not list_:
        return None
    if len_ == 1:
        return list_
    else:
        middle = len_ // 2
        left_l = merge(list_[0:middle])
        upper_l = merge(list_[middle:])
        i, j = 0, 0
        res = []
        while i < len(left_l) and j < len(upper_l):
            if left_l[i] < upper_l[j]:
                res.append(left_l[i])
                i += 1
            else:
                res.append(upper_l[j])
                j += 1
        res.extend(left_l[i:] if j == len(upper_l) else upper_l[j:])
        return res


def quick(list_: List) -> None:
    from random import randint

    def _quick(list_: List, lower, upper):
        # 一定要设置 lower大于upper
        if lower >= upper:
            return
        lower_upper = lower
        pivot_idx = randint(lower, upper)
        pivot_val = list_[pivot_idx]
        list_[pivot_idx], list_[upper] = list_[upper], list_[pivot_idx]
        for i in range(lower, upper):
            if list_[i] < pivot_val:
                list_[i], list_[lower_upper] = list_[lower_upper], list_[i]
                lower_upper += 1
        list_[lower_upper], list_[upper] = list_[upper], list_[lower_upper]
        _quick(list_, lower, lower_upper - 1)
        _quick(list_, lower_upper + 1, upper)

    _quick(list_, 0, len(list_) - 1)


def counting_sort(list_: list):
    max_ = max(list_)
    min_ = min(list_)
    bucket = [0]*(max_-min_+1)
    for e in list_:
        bucket[e-min_] += 1
    j = 0
    for i, count in enumerate(bucket):
        while count != 0:
            list_[j] = min_+i
            j += 1
            count -= 1


def counting_sort2(list_: list):
    max_ = max(list_)
    min_ = min(list_)
    bucket = [0] * (max_ - min_ + 1)
    new_list = [None] * len(list_)
    for e in list_:
        bucket[e - min_] += 1
    # accumulate count
    for i in range(1, len(bucket)):
        bucket[i] = bucket[i] + bucket[i - 1]
    for e in reversed(list_):
        new_list[bucket[e - min_] - 1] = e
        bucket[e - min_] -= 1
    for i in range(len(list_)):
        list_[i] = new_list[i]


def counting_sort4radix(list_: list, digits):
    def extract_dig(e):
        return (e // digits) % 10
    bucket = [0] * 10
    new_list = [None] * len(list_)
    for e in list_:
        d = extract_dig(e)
        bucket[d - 0] += 1
    # accumulate count
    for i in range(1, len(bucket)):
        bucket[i] = bucket[i] + bucket[i - 1]
    for e in reversed(list_):
        d = extract_dig(e)
        new_list[bucket[d - 0] - 1] = e
        bucket[d - 0] -= 1
    for i in range(len(list_)):
        list_[i] = new_list[i]


# Main function to implement radix sort
def radix(array):
    max_element = max(array)
    place = 1
    while max_element // place > 0:
        counting_sort4radix(array, place)
        place *= 10


def test(func, a, inplace=True):
    for p in itertools.permutations(a.copy()):
        p = list(p)
        if inplace:
            func(p)
            assert a == p, 'algorithms fails'
        else:
            assert a == func(p), 'algorithms fails'
    print('Test successful!')


def runtime(funcs,arguments):
    import time
    import random
    random.shuffle(arguments)
    # print(arguments,'\n')
    for func in funcs:
        s=time.process_time()
        func(arguments.copy())
        e=time.process_time()
        print(f'{func.__name__} sort running time is : {e-s}s.')


# x = list(range(2 ** 12))
# funcs=[bubble,selection,insertion]
# runtime(funcs,x)
# a=[5,6,7,4,3,2,1]
# insertion(a)
# print(a)

a = [1, 2, 3, 4, 5, 6]
# a.reverse()
# print(a)
# quick(a)
# print(a)

test(bubble, a)
