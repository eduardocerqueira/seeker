#date: 2021-11-19T17:12:27Z
#url: https://api.github.com/gists/9f8a2eaa4f91ba0745c08b2785027e5c
#owner: https://api.github.com/users/garthgilmour

class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def __str__(self):
        return f"{self.__name} of age {self.__age}"


p1 = Person("Jane", 30)
print(p1)

p1.__name = "Dave"
p1._Person__name = "Fred"

print(p1)
print(p1.__name)