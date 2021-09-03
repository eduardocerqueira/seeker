#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def f(x):
    kolvo = 2
    for j in range(2,int(x**0.5)+1):
        if x%j == 0:
            kolvo += 1
            if j*j != x:
                kolvo += 1
    if kolvo%2 == 1:
        return kolvo
    else:
        return False

counter = 0
a = 194441
b = 196500
for i in range(a,b+1):
    if f(i)%2 == 1:
        counter += 1
        print(counter,i,f(i),int(i**0.5))