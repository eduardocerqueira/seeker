#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

choslo = 10000000
#______ = 6 znakov

for x in range(5000000):

    q = x

    S = x

    R = 0

    while x > 0:

        d = x%2

        R = 10 * R + d

        x = x//2

    S = R + S

    S = str(S)

    dlina = len(S)

    if dlina == 6:
        print(S,dlina,q)

        if q < choslo:
            choslo = q

print(choslo)

#min = 33