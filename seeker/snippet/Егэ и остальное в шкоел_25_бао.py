#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def resh(x):
    a = [True]*(x+1)
    a[0] = False
    a[1] = False
    for i in range(2,int(x**0.5)+1):
        if a[i]:
            if i == 2:
                j = 2
                while j*i <= x:
                    a[j*i] = False
                    j += 1
            else:
                j = i
                while j*i <= x:
                    a[j*i] = False
                    j += 2
    return a



a = 3000000
b = 10000000

t = resh(b)
w = []
for i in range(len(t)):
    if t[i] == True:
        w.append(i)
counter = 0
maxim = 0
predmaxim = 0
for i in range(len(w)-1):
    if w[i+1] - w[i] == 2 and w[i] >= a:
        counter += 1
        predmaxim = w[i]
        maxim = w[i+1]
print(counter,(maxim + predmaxim)//2)