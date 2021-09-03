#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

k = int(input())
a, x = map(int, input().split())
b, y = map(int, input().split())
ans = 0

if a < b and x > y:
    k = k - a
    ans += k * x
    k = k - b
    ans += k * y
elif b < a and y > x:
    k = k - b
    ans += k * y
    k = k - a
    ans += k * x
else:
    ans1 = 0
    ans2 = 0
    q = k

    k = k - a
    ans1 += k * x
    k = k - b
    ans1 += k * y

    k = q

    k = k - b
    ans2 += k * y
    k = k - a
    ans2 += k * x

    if ans1 > ans2:
        ans = ans1
    else:
        ans = ans2
print(ans)
