#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

n = input()

counter = 0
maxim = 0

for i in range(len(n)-2):
    if n[i] == n[i+1] and n[i+1] != n[i+2]:
        counter += 1
        print(n[i])
    elif n[i] != n[i+1] and n[i+1] == n[i+2] and n[i] == n[i-1]:
        counter += 1
        print(n[i])
    else:
        print(n[i])
        print()
        print(counter)
        if counter+1 > maxim:
            maxim = counter+1
        counter = 0
if counter > maxim:
    maxim = counter

print(maxim)