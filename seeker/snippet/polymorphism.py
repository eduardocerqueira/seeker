#date: 2022-08-25T15:16:17Z
#url: https://api.github.com/gists/382a5b6433d8ae97e4de09489edc17c0
#owner: https://api.github.com/users/ayseaktag

class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print(f"I am a cat. My name is {self.name}. I am {self.age} years old.")

    def make_sound(self):
        print("Miyavv")


class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def info(self):
        print(f"I am a dog. My name is {self.name}. I am {self.age} years old.")

    def make_sound(self):
        print("Hav Hav")


cat1 = Cat("Missha", 7)
dog1 = Dog("Zeytin", 5)

for animal in (cat1, dog1):
    animal.make_sound()
    animal.info()
    animal.make_sound()