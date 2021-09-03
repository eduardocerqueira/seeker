#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):
    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            return False
    return True

a = []

for i in range(2,int(190072**0.5)+1):
    if simple(i) and i%2 == 1:
        a.append(i)

for h in range(len(a)):
    if a[h]**5 <= 190072 and a[h]**5 >= 190061:
        print(a[h])