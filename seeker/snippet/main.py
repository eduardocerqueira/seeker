#date: 2024-11-19T17:10:34Z
#url: https://api.github.com/gists/40b980eadde4a630c8fd8dae655bce4e
#owner: https://api.github.com/users/mypy-play

# mypy: enable-error-code="possibly-undefined"
def func(flag: bool) -> str:
    if flag:
        name = "Name"
    return name


func(False)