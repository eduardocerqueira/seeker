#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):

    for j in range(2,int(x**0.5)+1):
        if x%j == 0:
            return False

    return True

def abs_1(y):
    if a[i] - sred < 0:
        return -(a[i] - sred)

    if a[i] - sred > 0:
        return a[i] - sred

a = []


for i in range(2,int(305283**0.5)+1):
    if simple(i):
        a.append(i)


s = len(a)

counter = 0

for h in range(s):
    for c in range(h):
        for k in range(c):
            if (a[h]*a[c]*a[k]) < 305284 and (a[h]*a[c]*a[k]) > 236227:
                counter += 1

sred = sum(a)/s

sorted(a)

minim = 1000000000
minim_znach = 101010101010
for i in range(s):
    if abs_1(i) < minim:
        minim = abs_1(i)
        minim_znach = a[i]

print(a)



print(counter,minim_znach)