#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = int(input())

maxim = 0
minim = 100000000000000
minimnech = 100000000000000000


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

    if x > maxim:
        if maxim%2 == 1 and maxim < minimnech:
            minimnech = maxim
        maxim = x
    elif x < minimnech and x%2 == 1:
        minimnech = x

    if z < minim:
        if minim%2 == 1 and minim < minimnech:
            minimnech = minim
        minim = z
    elif z%2 == 1 and z < minimnech:
        minimnech = z

    if y%2 == 1 and y < minimnech:
        minimnech = y


print(minim,maxim,minimnech)
#1 10000 1