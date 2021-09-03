#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = int(input())

maxim = 0
minim = 0
ost = 0


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

    minim += z
    maxim += x
    ost += y
print(maxim,minim,ost)
#469845759 156726830 313756423