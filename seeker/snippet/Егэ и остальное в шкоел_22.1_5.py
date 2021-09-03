#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

maxim = 0
for x in range(-100,100):
    l = x - 15
    m = x + 15
    while l != m:
        if l > m:
            l = l - m
        else:
            m = m - l
        if l < 100000:
            break
    if m == 15:
        maxim = x
print(maxim)
#0