#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

for x in range(1,100000):

    a = 0

    q = x

    b = 10

    while x > 0:

        d = x % 9

        if d > a :
            a = d

        if d < b :
            b = d

        x = x//9

    if (a + b) == 11:
        print(a+b,q)

#min = 35