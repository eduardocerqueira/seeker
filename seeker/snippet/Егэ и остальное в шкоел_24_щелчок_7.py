#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

maxim = 0
counter = 0

for i in range(len(n)):
    if n[i] == "D":
        counter += 1
    else:
        if counter > maxim:
            maxim = counter
        counter = 0
if counter > maxim:
    maxim = counter
print(maxim)