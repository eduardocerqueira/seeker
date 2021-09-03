#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def deli(x):
    a = [1]
    for j in range(2,int(x**0.5)+1):
        if x%j == 0:
            a.append(j)
            if j*j != x:
                a.append(x//j)
    a.append(x)
    return a



def prima(x):
    t = int(x**0.5)+1
    a = [True]*t
    for i in range(2,t):
        for j in range(2,i):
                if i%j == 0:
                    a[i] = False
    a[0] = False
    a[1] = False
    return a

w = prima(196500)
e = []
for i in range(len(w)):
    if w[i] == True:
        e.append(i**2)

q = 194441
b = 196500
for i in range(len(e)):
    if e[i] > q and e[i] < b:
        print(deli(e[i]))

