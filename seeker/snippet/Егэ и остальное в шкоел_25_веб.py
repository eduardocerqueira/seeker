#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def F(n):
    counter = 0
    for i in range(2,n+1,2):
        if n%i==0:
            counter += 1

    if counter == 6:
        for i in range(2, n + 1, 2):
            if n % i == 0:
                print(i,end=" ")
        print()

a = 95632
b = 95700
counter = 0
for i in range(a,b+1):
    F(i)