#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558


def is_simple(y):
    for z in range(2,int(y**0.5)+1):
        if i%z == 0:
            return False
    return True



def kolvo_divisors(n):
    div = 0
    for j in range(2,int(n**0.5)+1):
        if i%j==0 and is_simple(j):
            div += 1
    return div



minim = 100000000000000
for i in range(432102,442110):
    if kolvo_divisors(i) == 4 and i%6!=0:
        if i < minim:
            minim = i
print(minim)

