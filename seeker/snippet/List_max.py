#date: 2022-06-20T17:07:51Z
#url: https://api.github.com/gists/8768717653a39f6674f1894e2331b508
#owner: https://api.github.com/users/HousniBouchen


liste = [5, 6, 2, 10, -12, 123, 1, 147, 128, 169, 5]
max=liste[0]

for i in range(1,len(liste)):
    if max<liste[i]:
        max=liste[i]

print(max)