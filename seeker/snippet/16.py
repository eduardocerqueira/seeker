#date: 2021-09-29T17:07:45Z
#url: https://api.github.com/gists/93402a04d52f61715f71f9ce90b4888c
#owner: https://api.github.com/users/makhokhrin

# 1 2 3 4 5 6 7  8
# 1 1 2 3 5 8 13 21
'''
def fib(n):
    if n == 1 or n == 2:
        return 1
    return fib(n - 1) + fib(n - 2)


n = int(input())
print(fib(n))


def f(a, n):
    if n == 1:
        return a
    if n % 2 == 0:
        return f(a, n // 2) * f(a, n // 2)
    else:
        return f(a, n - 1) * a


def f2(a, n):
    if n != 1:
        return f2(a, n - 1) * a
    return a


print(f(7, 1000))
print(f2(7, 999))
'''
'''
def F(n):
    if n == 1:
        return 1
    if n % 2 == 0:
        return n + F(n - 1)
    if n > 1 and n % 2 != 0:
        return 2 * F(n - 2)


print(F(24))
'''
'''
def F(n):
    if n == 0:
        return 0
    if n > 0 and n % 3 == 0:
        return F(n // 3)
    if n % 3 > 0:
        return n % 3 + F(n - n % 3)

for i in range(0, 1000):
    if F(i) == 11:
        print(i)
'''
