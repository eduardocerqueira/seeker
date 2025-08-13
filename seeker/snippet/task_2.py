#date: 2025-08-13T16:40:41Z
#url: https://api.github.com/gists/c16e305a437ee7787571bd8dbb39ee8c
#owner: https://api.github.com/users/DanielIvanov19

class Employee:
    def __init__(self, name, age):
        self.name = name
        self.age = age


def my_vars(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"{type(obj).__name__} няма __dict__")
