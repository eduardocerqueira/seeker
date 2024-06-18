#date: 2024-06-18T16:49:31Z
#url: https://api.github.com/gists/e2908dcedf57fb03e93b46cc6cb066a2
#owner: https://api.github.com/users/MaryBarani

# Homework: Classes
# Read carefully until the end before you start solving the exercises.

# Practice the Basics

# Basic Class

# - Create an empty class HouseForSale
# - Create two instances.
# - Add number_of_rooms and price as instance attributes.
# - Create print statements that show the attribute values for the instances.
class HouseForSale:
    pass


house_1 = HouseForSale()
house_2 = HouseForSale()
house_1.number_of_rooms = 4
house_2.number_of_rooms = 3

house_1.price = 850000
house_2.price = 930000

print(f"The first house has {house_1.number_of_rooms} rooms and costs ${house_1.price}.")
print(f"The second house has {house_2.number_of_rooms} rooms and costs ${house_2.price}.")


# ----------------------------------------------------------------------------------------------------------------------

# Instance Methods

# - Create a Computer class.
# - Create method:
#   - turn_on that prints Computer has Turned On
#   - turn_off that prints Computer has Turned Off
# - Create an instance of the Computer class then call each method.

class Computer:

    def turn_on(self):
        print(f"The computer has Turned On")

    def turn_off(self):
        print(f"The computer has Turned On")


computer = Computer()
computer.turn_on()
computer.turn_off()


# ----------------------------------------------------------------------------------------------------------------------

# Constructor with Parameters

# - Create a Dog class.
# - Dog should have a constructor with a name parameter.
# - Dog should have a method say_name that prints the name of the dog.

class Dog:

    def __init__(self, name):
        self.name = name

    def say_name(self):
        print(f"My name is {self.name}")


dog = Dog("Bamby")
dog.say_name()


# ----------------------------------------------------------------------------------------------------------------------

# Inheritance

# Create the classes that would make the following code possible, with the caveat that
# both Dog and Cat must inherit from Animal.

class Animal:
    def say_name(self):
        print("I don't have a name yet.")

    def speak(self):
        print("I can't speak!")


class Dog(Animal):

    def __init__(self, name):
        self.name = name

    def say_name(self):
        print(self.name)

    def speak(self):
        print("Woof!")


class Cat(Animal):

    def __init__(self, name):
        self.name = name

    def say_name(self):
        print(self.name)

    def speak(self):
        print("Meow!")


animal = Animal()
animal.say_name()  # Prints: I don't have a name yet.
animal.speak()  # Prints: I can't speak!

dog = Dog('Fido')
dog.say_name()  # Prints: Fido
dog.speak()  # Prints: Woof!

cat = Cat('Max')
cat.say_name()  # Prints: Max
cat.speak()  # Prints: Meow!

# ----------------------------------------------------------------------------------------------------------------------

# Exercises
# Exercise 1: Books and Authors

# Create an empty class called Book. Then create three instances.
# Add the following attributes for each of the instances: title, author, and publication_year.
# Create print statements to display the attributes of each one of the instances.

# Pre-code:
class Book:
    pass


book1 = Book()
book1.title = 'To Kill a Mockingbird'
book1.author = 'Harper Lee'
book1.publication_year = 1960

book2 = Book()
book2.title = 'Meeting in the dark'
book2.author = 'Maria whose'
book2.publication_year = 1821
# Your code here

print(f"The first book is {book1.title} written by {book1.author} in {book1.publication_year}.")
print(f"The second book is {book2.title} written by {book2.author} in {book2.publication_year}.")

# ----------------------------------------------------------------------------------------------------------------------

# Exercise 2. Vehicle and Types of Vehicles

# Create a Vehicle class.
# - Its constructor should take the name and type of the vehicle and store them as instance attributes.
# - This Vehicle class should also have a show_type() instance method that prints out the
#   message: "<NAME_OF_VEHICLE> is a <TYPE_OF_VEHICLE>"
# - Create Car and Bike classes that inherit from Vehicle.
# - Create instances of Car and Bike and make them show their types.


class Vehicle:
   def __init__(self, name, vehicle_type):
      self.name = name
      self.type = vehicle_type

   def show_type(self):
      print(f"{self.name} is a {self.type}")


class Car(Vehicle):
   pass


class Bike(Vehicle):
   pass


car_1 = Car("Jeep", "Car")

bike_1 = Bike("Raleigh", "Bike")

car_1.show_type()
bike_1.show_type()

# ----------------------------------------------------------------------------------------------------------------------

# Exercise 3. Spot and correct the mistakes

# - You are given a task to create a Car class.
# - Each car will have attributes for model and year.
# - Unfortunately, the given code below contains several mistakes.
# - Your task is to find and correct these mistakes to make the code run successfully.
# - Please include a comment in the code explaining the corrections you made and why.

# Pre-code
class Car:
    def __init__(self, model, year):  # the first argument of magic and instance methods should be self
        self.model = model  # self refers to the object not an attribute
        self.year = year  # to define an instance attribute we should use self.


my_car = Car("Toyota",
             2020)  # the constractor expect 2 arguments, so we should either pass 2 arguments or give a default
# value to the second argument in the class definition
print(my_car.model)
print(my_car.year)
# ----------------------------------------------------------------------------------------------------------------------

# Exercise 4. SmartHome with Constructor

# Create a SmartHome class that has a constructor __init__ and a send_notification() method.
#
# The constructor should initialize the attributes:
# - home_name
# - location
# - number_of_devices
#
# send_notification() should print a notification including the home_name and location.

# Create instances for the following:
# Home Name      Location      Number of Devices
# Villa Rosa     New York      15 devices
# Green House    California    10 devices
# Sea View       Florida       20 devices

# Call the send_notification() method for each instance,
# passing a message reminding to turn off the lights.

class SmartHome:
   def __init__(self, home_name, location, number_of_devices):
      self.home_name = home_name
      self.location = location
      self.number_of_devices = number_of_devices

   def send_notification(self, notification_message):
      print(f"{self.home_name}, {self.location}: {notification_message}")


home_1 = SmartHome("Villa Rosa", "New York", 15)
home_2 = SmartHome("VGreen House", "California", 10)
home_3 = SmartHome("Sea View", "New Florida", 20)

home_1.send_notification("turn off the lights")
home_2.send_notification("turn off the lights")
home_3.send_notification("turn off the lights")

# ----------------------------------------------------------------------------------------------------------------------

# Exercise 5. Inheritance. Spot and correct mistakes

# You should have the following hierarchy of classes:

# Animal
# │
# ├── Mammal
# │
# ├── Bird
# │
# └── Fish

# Each class has the following attributes:

# - Animal name
# - Mammal name, age, number of legs
# - Bird name, age, can fly or not
# - Fish name, age, number of fins

# But, the provided code for these classes and their instances has several mistakes
# related to hierarchy, class attributes, and instance creation.

# Find and correct these mistakes to make the code work properly.
# Leave a comment in the code explaining what the problems were and why it wouldn't work.
# There are seven mistakes in the pre-code.

# Pre-code
class Animal:
    def __init__(self, name, age):
        self.name = name  # To ensure that name is an instance attribute inherited by child classes, we should prefix
        # it with self.. Otherwise, name and age will be treated as local variables within the method's scope.
        self.age = age


class Mammal(Animal):  # the is no Animals class (the Animal is correct)
    def __init__(self, name, age, num_legs):
        super().__init__(name, age)
        self.num_legs = num_legs


class Bird(Animal):
    def __init__(self, can_fly, name="Bird", age=1):
        super().__init__(name, age) # To ensure that Bird instances inherit name and age from the parent class,
        # we need to call the constructor of the Animal class.
        self.can_fly = can_fly


class Fish(Animal): # Fish should inherit from Animal based on the above hierarchy of classes
    def __init__(self, name, age, num_fins):
        super().__init__(name, age)
        self.num_fins = num_fins


tiger = Mammal('Tiger', 5, 4)
sparrow = Bird(True) # or sparrow = Bird(True, 'Clow', 2)
goldfish = Fish('Goldfish', 2, 'Many')
