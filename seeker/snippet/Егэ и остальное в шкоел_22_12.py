#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

maxim = 0
# 4, 2

for x in range(1000000):

    q = x

    a = 0

    b = 0

    while x > 0:
        c = x%2

        if c == 0:

            a += 1

        else:
            b += 1

        x = x//10

    if a == 4 and b == 2:
        print(a, b , q)

        if q > maxim:
            maxim = q

print(maxim)

#max = 998888