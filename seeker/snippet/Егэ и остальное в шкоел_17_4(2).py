#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

minim = 10000000000000000000
counter = 0
for i in range(1045,8964):
    if (i % 5 == 0) and (i % 7 == 0) and (i % 11 != 0) and (i % 13 != 0) and (i % 17 != 0) and (i % 23 != 0):
        counter += 1
        if i < minim:
            minim = i
print(minim, counter)