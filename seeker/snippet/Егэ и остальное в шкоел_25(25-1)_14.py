#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def summa_div(x):
    summa = 1 + x
    for j in range(2,int(x**0.5)+1):
        if x%j==0:
            summa += j
    return summa

for i in range(194441,196501):
    if summa_div(i) > 290000:
        print(i)

#Снова краб?