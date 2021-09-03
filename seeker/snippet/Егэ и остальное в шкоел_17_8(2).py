#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

counter = 0
minim = 10000000
for i in range(3712,8433):
    if i%2 == i%4 and (i%13==0 or i%14==0 or i%14==0):
        counter += 1
        if i < minim:
            minim = i
print(minim, counter)