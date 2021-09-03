#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

counter = 0

for i in range(len(n)):
    if n[i] == "M" and n[i] == n[i+4]:
        counter += 1
print(counter)