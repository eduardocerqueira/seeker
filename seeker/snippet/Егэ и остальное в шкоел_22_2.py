#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

for x in range(10000):
    a = 0
    b = 1
    q = x

    while x > 0:
        if x % 2 > 0:
            a += x%12

        else:
            b *= x%12

        x = x//12

    if a == 2 and b == 10:
        print(a,b,q)
#min = 166