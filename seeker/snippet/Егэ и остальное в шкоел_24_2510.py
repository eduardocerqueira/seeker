#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

counter = 0
maxim = 0

for i in range(len(n)):
    if n[i] != "D" and n[i] != "E":
        counter += 1
    else:
        if maxim < counter:
            maxim = counter
        counter = 0
print(maxim)
# 16 +