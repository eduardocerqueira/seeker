#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

minim = 10000000000

for x in range(201,10000):

    q = x

    l = x - 23

    m = x + 23

    while l!=m:
        if l > m:
            l = l - m

        else:
            m = m - l

    if m == 23:
        if minim > q:
            minim = q
        print(m,q)

print(minim)

#min = 230