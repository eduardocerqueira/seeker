#date: 2022-12-08T16:57:08Z
#url: https://api.github.com/gists/f0b609bf5bc5bc131fac62d42e05cb2a
#owner: https://api.github.com/users/tidu54

from random import randint
a=randint(1,100)
b=-1
while b!=a:
    b=int(input("dit un nombre pour esayer de trouver"))
    if b<a:
        print("le nombre est plus grand")
    else:
        print("le nombre est plus petit")

print("bravo tu as réussi")
