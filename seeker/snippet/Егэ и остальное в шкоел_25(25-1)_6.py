#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):
    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            return False
    return True

for i in range(2,int(957812**0.5)+1):

    if simple(i):

        s = i**4

        if s > 152346 and s < 957812:

            print(i**3,s)