#date: 2023-04-21T16:50:14Z
#url: https://api.github.com/gists/cd63ba419dbf833519a3114de359a9e2
#owner: https://api.github.com/users/yoanymora

import math
# get a, b, c
print('Give me a')
a = int(input())
print('Give me b')
b = int(input())
print('Give me c')
c = int(input())

# compute s
s = 0.5 * (a + b + c)
print('s is ' + str(s))

# first formula
# compute r
r = (math.sqrt(s * (s - a) * (s - b) * (s - c))) / s
print('r is equal to ' + str(r))
# a comparison I don't understand :s
# if (r > ?):
#     print("El punto esta fuera del circulo.")
# else if r == ?:
#     print("El punto esta fuera del circulo.")
# else:
#     print("El punto esta dentro del circulo.")

# second formula
# compute R
R = (a * b * c) / (4 * (math.sqrt(s * (s - a) * (s - b) * (s - c))))
print('R is equal to ' + str(R))
