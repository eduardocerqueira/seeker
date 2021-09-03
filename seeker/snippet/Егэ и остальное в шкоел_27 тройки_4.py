#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = int(input())

maxim = 0
ch = 0
nech = 0
mindiff = 0
predmindiff = 0
for i in range(n):
    x,y,z = map(int,input().split())

    if x < y:
        t = y
        y = x
        x = t
    if y < z:
        t = z
        z = y
        y = t
    if x < y:
        t = y
        y = x
        x = t

    maxim += x
    ch += y
    nech += z

    if x != y:
        diff = x - y
    else:
        diff = y - z

    if diff >= mindiff and diff%2 != 0 and diff != 0:
        predmindiff = diff
        if predmindiff >= mindiff:
            t = mindiff
            mindiff = predmindiff
            predmindiff = t

if (ch%2 == 0 or nech%2 == 0) and (ch%2 == 1 or nech%2 == 1):
    print(maxim)
elif ch%2 == 0 and nech%2 == 0:
    print(maxim - mindiff)
elif ch%2 == 1 and nech%2 == 1:
    print(maxim - mindiff)