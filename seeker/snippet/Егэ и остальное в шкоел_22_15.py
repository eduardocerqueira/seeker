#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

minim = 1000000000
#_____ = 5 znakov

for x in range(1000000):
    S = x

    R = 0

    q = x

    while x > 0:

        d = x%2

        R = 10*R + d

        x = x//2

    S = R + S

    S = str(S)

    dlina = len(S)

    if dlina == 5:
        print(dlina,S,q)

        if minim > q:
            minim = q

print(minim)

#min = 17