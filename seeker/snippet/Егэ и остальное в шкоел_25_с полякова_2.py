#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def prime(x):
    for i in range(2,int(x**0.5)+1):
        if x%i == 0:
            return False
    return True

def resheto(x):
    a = [True]*(x)
    a[0] = False
    a[1] = False
    b = []
    for i in range(2,x):
        if prime(i) and i%2 == 1:
            continue
        else:
            a[i] = False
    for i in range(x):
        if a[i] == True:
            b.append(i)
    return b

def k(x):
    kolvo = 1
    if x%2 == 1:
        kolvo += 1
    for j in range(2,int(x**0.5)+1):
        if x%j == 0 and j%2 == 1:
            kolvo += 1
            if x//j != j and (x//j)%2 == 1:
                kolvo += 1
        if x // j != j and (x // j) % 2 == 1 and x%j == 0 and j%2 == 0:
            kolvo += 1
        if kolvo > 5:
            return False
    if kolvo != 5:
        return False
    elif kolvo == 5:
        return True
h = int(60000000**0.5)+1

t = resheto(h)
zxc = []
for j in range(len(t)):
    for x in range(1,10):
        g = (2**x)*j**4
        if g >= 55000000 and g <= 60000000:
            print(g)
            zxc.append(g)


for i in range(2,h):
    c = i**2
    if c >= 55000000 and c <= 60000000:
        if k(c):
            print(c)
zxc.sort()

for i in range(len(zxc)):
    for j in range(i+1,len(zxc)):
        if zxc[i] == zxc[j]:
            zxc[i] = 0
zxc.sort()
print(zxc)