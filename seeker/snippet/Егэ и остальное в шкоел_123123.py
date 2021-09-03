#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = int(input())
f = open("../ntrr")

s = 0
m = [1000000000000]*73
l = [-1]*73
m[0] = 0
l[0] = -1
maxim = 0
ans = 0
for i in range(n):
    x = int(f.readline())
    s = s + x
    ost = s % 73
    if (s - m[ost] > maxim):
        maxim = s - m[ost]
        ans = i - l[ost]
    if (s < m[ost]):
        m[ost] = s
        l[ost] = i
print(ans)