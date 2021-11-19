#date: 2021-11-19T17:09:50Z
#url: https://api.github.com/gists/3b2d576cfc06edbb8cd45076ca3d44cc
#owner: https://api.github.com/users/garthgilmour

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} of age {self.age}"