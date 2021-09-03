#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

minim = 1000000000
# 4, 11

for x in range(1000000):

    a = 0

    q = x

    b = 1

    while x > 0:
        if x % 2 > 0:
            a += x%13

        else:

            b = b * (x%13)

        x = x //13

    if a == 4 and b == 11:

        if minim > q:
            minim = q

        print(a, b , q)

print(minim)

#min = 315