#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

for x in range(96,100000):
    q = x
    l = x
    m = 51
    if l%2 == 0:
        m = 36

    while l!=m:
        if l > m:
            l = l - m
        else:
            m = m - l

    if m == 18:
        print(q,m)
        break
#126