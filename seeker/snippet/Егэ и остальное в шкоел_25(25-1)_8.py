#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):

    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            return False

    return True


a = [0]*10

b = []

for i in range(2,int(309877**0.5)):
    if simple(i):
        b.append(i)

s = len(b)


maxim = 0


for h in range(s):

    for c in range(h):

        if (b[h]*b[c]) < 309877 and (b[h]*b[c]) > 237981:

            if (b[h]*b[c]) > maxim:
                maxim = b[h]*b[c]

            a[(b[h]*b[c])%10] += 1


print(a,maxim)