#date: 2024-04-29T17:06:09Z
#url: https://api.github.com/gists/2878749d8b984e34d5eb5527d7fe9198
#owner: https://api.github.com/users/mypy-play

x: str | int = 42

if bool():
    x = 'foo'
elif bool():
    x = 'bar'
else:
    x = 'baz'

reveal_type(x)