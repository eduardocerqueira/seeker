#date: 2024-05-06T17:04:33Z
#url: https://api.github.com/gists/e73c666ecf44234a050d512c6e669441
#owner: https://api.github.com/users/CodeAnge1

def f(c,e):
    if c>e: return 0
    if c==e: return 1
    return f(c+1, e)+f(c+2, e)+f(c*2, e)


print(f(4,11)*f(11,13)*f(13,15))