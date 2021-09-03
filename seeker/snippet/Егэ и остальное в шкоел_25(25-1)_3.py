#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def simple(x):

    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            return False
    return True


a = []
for i in range(2,int(363250**0.5)+1):
    if simple(i):
        a.append(i)

a.sort()

s = len(a)

b = []

for i in range(s+1):
    for j in range(i,s-1):
        if a[i]*a[j] > 298434 and a[i]*a[j] < 363250:
            b.append(a[i]*a[j])
b.sort()

o1 = len(b)

sr = sum(b) / len(b)

if ((b[o1//2] - sr) < (b[(o1//2)+1] - sr) and (b[o1//2] - sr < b[(o1//2)-1])):
    o2 = b[o1//2]
else:
    if b[o1//2+1] - sr < b[o1//2-1] - sr:
        o2 = b[o1//2+1]
    else:
        o2 = b[o1//2-1]

print(o1,o2)
print(b)
