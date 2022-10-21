#date: 2022-10-21T17:20:42Z
#url: https://api.github.com/gists/44466a079542d91104366632c881039e
#owner: https://api.github.com/users/saintdianabless

from collections import Counter
import itertools
from typing import List


def err(msg: str):
    print(msg)
    exit(1)


def parse_number_str(numbers: str) -> Counter:
    try:
        numbers = numbers.split(',')
        # numbers = map(lambda num: num.strip(' '), numbers)
        numbers = map(int, numbers)
        return Counter(numbers)
    except Exception as e:
        err(e)


def parse_ops_str(ops: str) -> Counter:
    ops = ops.split(',')
    ops = map(str.strip, ops)
    return Counter(ops)


def transform(counter: Counter, max_valid):
    res = []
    for k, v in counter.items():
        fill = v
        if fill > max_valid:
            fill = max_valid
        res += [k] * fill
    return res


def op(a, op, b):
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '**':
        return a**b
    if op == '//':
        return a // b
    if op == '%':
        return a % b
    if op == '|':
        return a | b
    if op == '&':
        return a & b
    if op == '^':
        return a ^ b
    if op == '>>':
        return a >> b
    if op == '<<':
        return a << b

    err('未知运算符: ' + op)


def calc(nums, ops):
    res = nums[0]
    for i in range(3):
        res = op(res, ops[i], nums[i + 1])
    return res


def perm(elements, size):
    perms = itertools.permutations(elements, size)
    return set(perms)


def how1024(numbers, ops) -> List[str]:
    numbers = parse_number_str(numbers)
    if sum(numbers.values()) < 4:
        err('至少4个数字')
    ops = parse_ops_str(ops)
    if sum(ops.values()) < 3:
        err('至少三个运算符')

    numbers = transform(numbers, 4)
    ops = transform(ops, 3)

    numperms = perm(numbers, 4)
    opperms = perm(ops, 3)

    result = []

    for numperm in numperms:
        for opperm in opperms:
            if calc(numperm, opperm) == 1024:
                result.append((numperm, opperm))
    
    return result


if __name__ == '__main__':
    while True:
        nums = input('输入数字, 逗号分隔, q退出: ')
        if nums == 'q':
            exit(0)
        ops = input('输入运算符, 逗号分隔: ')
        for r in how1024(nums, ops):
            print(r)
