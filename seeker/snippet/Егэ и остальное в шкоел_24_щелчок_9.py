#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

maxim = 0
maxi = 0
counter = 0
coun = 0

for i in range(len(n)-1):
    if n[i] != n[i+1]:
        coun += 1
    else:
        if coun+1 > maxi:
            maxi = coun+1
        coun = 0
if coun + 1 > maxi:
    maxi = coun + 1
for i in range(len(n) - 1):
    if n[i] == "D":
        counter += 1
    else:
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter
for i in range(len(n) - 1):
    if n[i] == "M":
        counter += 1
    else:
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter
for i in range(len(n) - 1):
    if n[i] == "V":
        counter += 1
        print(counter)
    else:
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter

print(maxi,maxim)