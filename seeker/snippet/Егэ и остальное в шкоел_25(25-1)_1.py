#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def kolvo_div(x):
    div = 2
    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            if j*j!=x:
                div += 2
            else:
                div += 1
    return div




for i in range(163700,164353):
    if kolvo_div(i) == 6:

        counter = 0
        a = []

        a.append(i)

        for h in range(2,int(i**0.5)):

            if i%h==0:

                a.append(i//h)

                break

        a.sort()
        print(i,a)