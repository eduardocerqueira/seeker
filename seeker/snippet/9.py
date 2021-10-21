#date: 2021-10-21T17:11:01Z
#url: https://api.github.com/gists/4ce45a8ea276e6c413990d8a41509f9f
#owner: https://api.github.com/users/kirilltobola

eps = 10 ** -12
a, b, c, d = map(float, input().split())

if abs((a+b) - (c+d)) < eps:
    print('equals')
elif (a+b) - (c+d) > eps:
    print('a+b greater')
elif (a+b) - (c+d) < eps:
    print('c+d greater')

print(a+b, c+d, (a+b) - (c+d))
#10.2 4.3 4.3 10.2
