#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def pros(x):
    t = int(x**0.5)+1
    c = [True]*t
    c[0] = False
    c[1] = False
    for j in range(2,t):
        for k in range(2,j):
            if j%k == 0:
                c[j] = False
    return c

a = 1000000
b = 2000000

w = pros(b)
s = []
for j in range(len(w)):
    if w[j] == True:
        s.append(j**6)

for j in range(len(s)):
    if s[j] >a and s[j] < b:
        print(s[j])
