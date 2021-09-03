#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def div(x):

    div = 2

    for j in range(2,int(x**0.5)+1):

        if x%j==0:

            if j*j == x:
                div += 1

            else:

                div += 2

    return div

a = []
counter = 0
for i in range(904528,997439):
    if div(i) == 5:
        a.append(i)
        counter += 1

a.sort()
s = len(a)


print(counter,a[s-1])