#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def kolvo_div(x):

    div = 0
    if x%2==0:
        div += 1
    for j in range(2,int(x**0.5)+1):
        if x%j == 0 and j%2 == 0:
            div += 1
            if (x/j)%2 == 0 and j*j!=x:
                div += 1

        elif x%j == 0 and j%2==1:
            if (x/j)%2 == 0:
                div += 1
    return div

a = []

for i in range(333555,778000):
    if kolvo_div(i) == 7:
        a.append(i)
print(a)