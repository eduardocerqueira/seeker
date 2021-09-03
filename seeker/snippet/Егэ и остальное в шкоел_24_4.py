#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

counter = 0

c = 0

maxim = 0


while counter < 216350:

    if n[counter] == "V":
        c += 1
        if c > maxim:
            maxim = c
    if n[counter + 1] != "V" and n[counter + 2] != "V":
        c = 0
    counter += 1

print(maxim-1)
# ans = 4