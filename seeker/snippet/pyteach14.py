#date: 2021-11-19T17:10:52Z
#url: https://api.github.com/gists/aded25358be1b9da8f2375257489c0e4
#owner: https://api.github.com/users/garthgilmour

class Person:
    def __init__(me, name, age):
        me.name = name
        me.age = age

    def __str__(me):
        return f"{me.name} of age {me.age}"
