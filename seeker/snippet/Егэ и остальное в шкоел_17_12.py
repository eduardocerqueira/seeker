#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558



def is_simple(y):
    if y == 1:
        return False
    for z in range(2,int(y**0.5)+1):
        if y%z == 0:
            return False
    return True


def kolvo_divisors(n):
    div = 0
    for j in range(1,int(n**0.5)+1):
        if n%j==0 and is_simple(j):
            div += 1
    return div


counter = 0
minim = 10000000000
for i in range(16807,279936):
    if i%6!=0 and kolvo_divisors(i) == 6:
        counter += 1
        if i < minim:
            minim = i
print(counter, minim)