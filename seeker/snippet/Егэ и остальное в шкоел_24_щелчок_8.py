#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

counter = 0
maxim = 0
q = 0
while q < len(n):
    if n[q] == "D":
        counter += 1
        q += 3
    else:
        q += 3
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter
q = 1
while q < len(n):
    if n[q] == "D":
        counter += 1
        q += 3
    else:
        q += 3
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter
q = 2
while q < len(n):
    if n[q] == "D":
        counter += 1
        q += 3
    else:
        q += 3
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter
print(maxim)
