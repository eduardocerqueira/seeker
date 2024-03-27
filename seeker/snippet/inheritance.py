#date: 2024-03-27T17:06:11Z
#url: https://api.github.com/gists/cbcab9f24eaa5c169194e4838db3c56f
#owner: https://api.github.com/users/JamesMenetrey

from abc import abstractmethod, abstractproperty, ABC
from typing import List


def load_from_file(filename):
    return [GuineaPig("Kéké"), GuineaPig("Choki"), Cat("Gaspard")]









class Animal(ABC):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def say_hello(self):
        pass


class Shop:
    def __init__(self, animals: List[Animal]):
        self.__animals = animals

    def hello_from_all(self):
        for animal in self.__animals:
            animal.say_hello()


class GuineaPig(Animal):
    def __init__(self, name):
        super().__init__(name)

    def say_hello(self):
        print(f"{self._name} said: Uiiik")


class Cat(Animal):
    def __init__(self, name):
        super().__init__(name)

    def say_hello(self):
        print(f"{self._name} said: Moew!")


my_shop = Shop(load_from_file("my_shop.txt"))
my_shop.hello_from_all()