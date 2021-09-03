#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def is_simple(x):
    for j in range(2,int(x**0.5)+1):
        if i%j==0:
            return False
    return True

def coun_divisirs(y):
    div = 0
    for n in range(2,int(y**0.5)+1):
        if i%n==0 and is_simple(n):
            div += 1
    return div

counter = 0
for i in range(44000,58001):
    if coun_divisirs(i) == 3:
        counter += 1
print(counter)
