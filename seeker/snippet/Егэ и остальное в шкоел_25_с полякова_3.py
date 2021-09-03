#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

a = []
b = []
for i in range(10000):
    if i % 2 == 0:
        for j in range(10000):
            if j % 2 == 1:
                t = (2 ** j) * (7 ** i)
                if t >= 100000000 and t <= 300000000:
                    a.append(t)
                    print(t,i+j)
                if t > 3000000000000000:
                    break
a.sort()
print(a)