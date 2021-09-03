#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558


a = []
c = []
maxim = 0
counter = 0
for i in range(1,5001):
    a.append(i)
    c.append(i**2)

for i in range(5000):
    for j in range(i,5000):
        print(counter)
        c1 = (a[i]**2 + a[j]**2)
        if c1 in c:
            counter += 1

print(counter)



