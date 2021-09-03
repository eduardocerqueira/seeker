#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

maxim = 0

for x in range(100000):

    q = x

    l = 0

    m = 0

    while x > 0:

        m += 1

        if x % 2 != 0:
            l = l + x%8

        x = x//8

    if l == 2 and m == 3:
        if maxim < q:
            maxim = q
        print(l,m,q)

print(maxim)

#max = 393
