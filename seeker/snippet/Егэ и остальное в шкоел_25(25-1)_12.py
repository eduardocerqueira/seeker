#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):

    for j in range(2,int(x**0.5)+1):
        if x%j == 0:
            return False
    return True

a = []

for i in range(4409962,4410102):
    if simple(i):
        a.append(i)

a.sort()

for h in range(len(a)):
    print(h," ",a[h])
