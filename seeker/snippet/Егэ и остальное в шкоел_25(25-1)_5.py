#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def div(x):

    div = 0

    if x%2==1:
        div += 1

    for j in range(2,int(x**0.5)+1):
        if j%2 == 1:

            if j*j == x and j%2==1:
                div += 1

            elif (x/j)%2 == 1 and j%2==1:
                div += 2

            elif j%2==1:
                div += 1
    return div

for i in range(190061,190073):

    if div(i) == 4:

        counter = 0

        a = []

        for h in reversed(range(2,i)):
            if i%h == 0 and h%2==1:
                a.append(h)
                counter += 1
                if counter == 2:
                    break
        print(i,a)
