#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

minim =1000000000000
counter = 0


for i in range(1170,8368):

    if (i%3 == 0 or i%7 == 0) and i%11!=0 and i%13!=0 and i%17!=0 and i%19!=0:

        if i < minim:
            minim = i
        counter += 1


if counter > minim:
    print(minim,counter)
else:
    print(counter,minim)
