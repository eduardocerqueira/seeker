#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = int(input())

per = 0
pr = 0
otvet = 0
min_diff = 1000000000000000000

for i in range(n):
    x,y,z = map(int,input().split())

    a = [x,y,z]
    a.sort()

    otvet += a[2]
    per += a[1]
    pr += a[0]

    if a[2] - a[1] != 0:
        diff = a[2] - a[1]
    elif a[2] - a[0] != 0:
        diff = a[2] - a[0]
    else:
        diff = 0

    if diff != 0 and diff%2 == 1 and min_diff > diff:
        min_diff = diff

if per%2 == 0 and pr%2 == 1:
    print(otvet)
elif per%2 == 1 and pr%2 == 0:
    print(otvet)
else:
    print(otvet-min_diff)
    #72345