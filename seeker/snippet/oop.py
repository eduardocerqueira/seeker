#date: 2023-03-30T16:48:03Z
#url: https://api.github.com/gists/051638c5943fb194cf72fca6c626ad4f
#owner: https://api.github.com/users/kallyas

class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        print("Generic animal sound..!")


class Dog(Animal):
    def __init__(self, name, age):
        self.age = age
        super().__init__(name)

    def make_sound(self):
        print(f"{self.name} is barking!")


dog = Dog(name="Doe", age=12)
dog.make_sound()


class Shape:
    def area(self):
        pass


class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
        self.PI = 3.14

    def area(self):
        return self.PI * (self.radius * self.radius)


class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width


class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return self.base * self.height * 0.5


sq = Square(side=2)
c1 = Circle(radius=4)
Rec = Rectangle(length=10, width=5)
Tri = Triangle(base=6, height=5)

print(sq.area())
print(c1.area())
print(Rec.area())
print(Tri.area())


class BankAccount:
    def __init__(self, bal):
        self.__bal = bal

    def deposit(self, amt):
        self.__bal += amt

    def withdraw(self, amt):
        if amt > self.__bal:
            raise ValueError("Insufficient Balance!..")
        self.__bal -= amt

    def get_balance(self):
        return self.__bal


acc = BankAccount(bal=1000)
print(acc.get_balance())
acc.withdraw(500)

try:
    acc.withdraw(600)  # Error!
except ValueError as e:
    print(e)

acc.deposit(200)
print(acc.get_balance())


class Vehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model

    def start(self):
        pass


class Car(Vehicle):
    def start(self):
        print("Starting the car...")


class MotorBike(Vehicle):
    def start(self):
        print("Starting a bike...")


car = Car(make="Vitz", model="Ford")
bk = MotorBike(make="Toyota", model="Ducati")
car.start()
bk.start()


# static method
class MathUtils:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @staticmethod
    def add(x, y):
        return x + y


print(MathUtils.add(2, 8))
