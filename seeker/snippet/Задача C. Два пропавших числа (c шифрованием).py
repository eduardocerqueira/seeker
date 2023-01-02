#date: 2023-01-02T16:39:06Z
#url: https://api.github.com/gists/4796e91539f457c4392bda506c234682
#owner: https://api.github.com/users/AspirantDrago

def first():
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    result = 0
    for x in arr:
        result |= 1 << x
    lst = []
    while result:
        lst.append(result & 63 + 32)
        result >>= 6
    print(''.join(map(chr, lst)))


def second():
    lft = list(input())
    n, k = map(int, input().split())
    arr = set(map(int, input().split()))
    result = 0
    while lft:
        result <<= 6
        result |= ord(lft.pop()) - 32
    for i in range(1, n + 1):
        if result & (1 << i):
            arr.add(i)
    answer = set(range(1, n + 1)) - arr
    print(*answer)


if input() == 'first':
    first()
else:
    second()


'''
first
15 3
4 6 1 10 11 12 13 14 15


second
RQ
15 4
5 2 9 8




first
9 3
4 6 1

second
R
9 4
5 2 9 8

'''