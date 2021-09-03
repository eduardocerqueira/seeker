#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

for x in range(10000):
    q = x

    l = 0

    m = 0

    while x > 0:
        m += 1
        if x % 2 != 0:
            l += 1

        x = x//2

    if l == 3 and m == 7:
        print(l,m,q)

#max = 112