#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def prime(x):
    for i in range(2,int(x**0.5)+1):
        if x%i == 0:
            return False
    return True

a = 100000000
b = 100000000000

for i in range(a,b+1):
    if prime(i):
        print(i)
