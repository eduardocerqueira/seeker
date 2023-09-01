#date: 2023-09-01T17:06:09Z
#url: https://api.github.com/gists/04a3192eecc77928ea6aa60f40604ef0
#owner: https://api.github.com/users/mypy-play

def func() -> 'map[int]':
    return map(int, '123')
    
m = func()
reveal_type(m)  # N: Revealed type is "builtins.map[builtins.int]"
reveal_type(next(m))  # N: Revealed type is "builtins.int"