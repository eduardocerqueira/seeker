#date: 2022-09-14T17:10:20Z
#url: https://api.github.com/gists/c308b409704ad82d197a7dbb23560a53
#owner: https://api.github.com/users/purplemonckey


from math import *

def f(x):
    a=(x+3)**2
    b=-a
    c=b+4
    return c


def g(x):
    a=x+1
    b=x+5
    c=-a*b
    return c

print('f(0)=',f(0))
print('f(-2)=',f(-2))
print('f(1)=',f(1))

print('g(0)=',g(0))
print('g(-2)=',g(-2))
print('g(1)=',g(1))
