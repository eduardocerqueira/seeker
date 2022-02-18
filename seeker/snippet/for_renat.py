#date: 2022-02-18T17:05:53Z
#url: https://api.github.com/gists/098d9861164c85cd5eed521d7be27c6d
#owner: https://api.github.com/users/Andrey-Bedretdinov

'''# ((x → y ) ≡ (z → w)) ∨ (x ∧ w)

print('x y z w')
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            for w in [0, 1]:
                if (((not x or y) == (not z or w)) or (x and w)) == 0:
                    print(x, y, z, w)


s = 0
k = 1
while s < 66:
    k += 3
    s += k
print(k)


def f(a: str):
    while '999' in a or '888' in a:
        if '888' in a:
            a = a.replace('888', '9', 1)
        else:
            a = a.replace('999', '8', 1)
    return a

print(f('9' * 1000))


def f(a):
    for x in range(1, 1000):
        for y in range(1, 1000):
            if not (2 * x + 3 * y < 30 or x * y >= a):
                return False
    return True


for x in range(10000000, 0, -1):
    if f(x):
        print(x)
        break


n = bin(4**2020 + 2**2017 - 15)[2:]
print(n.count('1'))


def F(n):
    if n > 0:
        F(n // 4)

        F(n - 1)

F(5)

with open('17-2.txt', 'r') as file:
    data = []
    for line in file:
        data.append(int(line))

even = []
for i in data:
    if i % 2 == 0:
        even.append(i)

arif = sum(even) / len(even)
print(arif)
count = 0
max_sum = 0
for i in range(len(data) - 1):
    a, b = data[i], data[i + 1]
    if (a % 3 == 0 or b % 3 == 0) and (a < arif or b < arif):
        count += 1
        max_sum = max(max_sum, a + b)

print(count, max_sum)


from functools import lru_cache


def moves(p):
    a, b = p
    return (a + 1, b), (a, b + 1), (a * 2, b), (a, b * 3)


@lru_cache(None)
def f(p):
    if sum(p) >= 69:
        return 'WIN'
    if any(f(x) == 'WIN' for x in moves(p)):
        return 'В1'
    if all(f(x) == 'В1' for x in moves(p)):
        return 'П1'
    if any(f(x) == 'П1' for x in moves(p)):
        return 'В2'
    if all(f(x) == 'В1' or f(x) == 'В2' for x in moves(p)):
        return 'П2'


print(((69 - 10) // 3 + 1) // 3 + 1)

ans = []
for s in range(10, 60):
    if f((10, s)) == 'В2':
        ans.append(s)
print(*ans)

for s in range(10, 60):
    if f((10, s)) == 'П2':
        print(s)
        break


def f(x):
    l = 0
    m = 0
    while x > 0:
        l += 1
        if (x % 2) != 0:
            m += x % 8
        x //= 8
    return l, m


for i in range(1000000, 0, -1):
    if f(i) == (3, 6):
        print(i)
        break


def f(start, stop):
    if start > stop:
        return 0
    if start == stop:
        return 1
    return f(start + 1, stop) + f(start + 2, stop) + f(start * 2, stop)

print(f(3, 10) * f(10, 12))


def divs(a):
    ans = set()
    for i in range(1, int(a ** 0.5 + 2)):
        if len(ans) > 5:
            return False
        if a % i == 0:
            if i % 2 == 1:
                ans.add(i)
            if (a // i) % 2 == 1:
                ans.add(a // i)
    if len(ans) == 5:
        return True


print(int('0110', 2), int('0101', 2), int('0100', 2), int('0011', 2))
'''


with open('24.txt', 'r') as file:
    data: str = file.read()

max_count = 0
count = 0
for i in range(len(data) - 3):
    a1, a2, a3, a4 = data[i], data[i + 1], data[i + 2], data[i + 3]
    if a1 + a2 + a3 + a4 != 'XZZY':
        if count == 0:
            count = 4
        else:
            count += 1
        max_count = max(max_count, count)
        continue
    count = 0
print(max_count)
