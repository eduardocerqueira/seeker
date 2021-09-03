#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = int(input())

maxim = 0
minim = 100000000000000000000000
predmch = 0



for i in range(n):
    x, y, z = map(int,input().split())

    if x < y:
        t = x
        x = y
        y = t
    if x < z:
        t = z
        z = x
        x = t
    if y < z:
        t = z
        z = y
        y = t

    if x > maxim:
        maxim = x
    elif x > predmch and x%2==0:
        predmch = x

    if y > predmch and y%2 == 0:
        predmch = y

    if z < minim:
        minim = z
    elif z > predmch and z%2 == 0:
        predmch = z
print(maxim,minim,predmch)
# 10000 1 10000