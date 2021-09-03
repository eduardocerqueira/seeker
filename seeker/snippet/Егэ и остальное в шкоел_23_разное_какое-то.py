#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def prost(x):
    for j in range(2,int(x**0.5)+1):
        if x%j == 0:
            return False
    return True

a = [0]*100
a[2] = 1
predprost = 2
for i in range(3,75):
    if i == 34:
        a[33] = 0
    a[i] += a[i-2]

    if prost(i):
        for k in range(predprost,i):
            a[i] += a[k]
            predprost = i

    if i == 14:
        for z in range(14):
            a[z] = 0
print(a[45])