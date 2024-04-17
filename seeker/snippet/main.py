#date: 2024-04-17T17:01:50Z
#url: https://api.github.com/gists/d83bae23d55fe9e27dcb4a5309f4b5c2
#owner: https://api.github.com/users/mypy-play

def identity(x):
    return x

y = identity(1)

reveal_type(y)