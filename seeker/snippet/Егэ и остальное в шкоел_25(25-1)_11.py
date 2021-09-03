#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):

    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            return False
    return True

a = []

for i in range(2,int(152673836**0.5)+1):
    if simple(i):
        a.append(i)

a.sort()

s = len(a)

b = []
c = []

for x in range(s):
    if a[x]**6 >= 106732567 and a[x]**6 <= 152673836:
        b.append(a[x]**6)
        c .append(a[x]**5)

for z in range(len(b)):
    print("  число    -  его делитель максимальный делитель   ")
    print(b[z]," - ",c[z])
