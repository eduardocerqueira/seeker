#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def summa(x):
    s = 0
    for j in range(2, int(x ** 0.5)+1):
        if i % j == 0:
            s += j + (i // j)
    return s


def counters(y):
    div = 0
    for c in range(2, int(y ** 0.5)+1):
        if i % c == 0:
            if c * c != i:
                div += 2
            else:
                div += 1
    return div


for i in range(115790, 143229):
    if summa(i) > 390000:
        print(summa(i), counters(i))
