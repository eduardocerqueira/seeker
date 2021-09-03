#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

a = [0]*100
a[10] = 1

for i in range(11,50):
    a[i] = a[i-2] + a[i-4] + a[i-3]
print(a[16])
# ans == 4