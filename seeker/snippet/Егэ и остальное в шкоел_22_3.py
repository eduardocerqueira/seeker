#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

for x in range(10000):
    l = 0

    q = x

    m = 0

    while x > 0:
        m = m+1

        if x % 2 !=0:
            l = l+1

        x = x//2

    if l == 5 and m == 6:
        print(l,m,q)
#min = 47