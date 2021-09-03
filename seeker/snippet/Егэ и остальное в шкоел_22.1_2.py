#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

for x in range(101,1000):
    l = x - 20
    m = x + 20
    q = x

    while l!=m:
        if l > m:
            l = l - m
        else:
            m = m - l

    if m == 20:
        print(q,m)
        break
#120