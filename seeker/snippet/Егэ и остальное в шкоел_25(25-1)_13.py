#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

a = []
i = 0
while 2**i <= 684934:
    if 2**i <= 684934 and 2**i >= 631632:
        a.append(2**i)
    i += 1
a.sort()

print(a)
if a == []:
    print("краб пойман!")