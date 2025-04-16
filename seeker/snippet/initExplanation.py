#date: 2025-04-16T16:54:56Z
#url: https://api.github.com/gists/7f66c8778ede8cb7c3c27049583185cc
#owner: https://api.github.com/users/Amarabolive

# You can ignore this. It's just so I can use print like it is in Python 3
from __future__ import print_function

# This is an object. This is one of Python's most basic types and what all classes should inherit from eventually.
# So even if your class inherits from another class which in turn inherits from another class, eventually if you follow it all the way back it should inherit from object
myObject = object()

# Object has a method called __init__
print(myObject.__init__)  # <method-wrapper '__init__' of object object at 0x000001F746C1F0A0>

# Therefore when you make a class it always has __init__ defined whether you intend for it or not
class Simple(object):
	pass # The pass keyword means nothing will be defined here

simple = Simple()

# As you can see this has an __init__ as well
print(simple.__init__)  # <method-wrapper '__init__' of Simple object at 0x000001F74745AC18>

# We can define our own __init__ however
class CustomInit(object):
	def __init__(self):
		print("Hello")

# And this gets run whenever we create an instance of our class
customInit = CustomInit() # Hello

# A class is what describes the object's behavior, and the instance is what we call the object after we run our class creator.
# Human could be a class, but Dave and Angela would be instances of a human
# For example, CustomInit is a class, but CustomInit() with the parenthesis gives us an instance of the class

# Running the __init__ is useful for declaring instance specific behavior or variables
# For example lets make a name

class Human(object):
	def __init__(self):
		self.name = None

# Lets make a few humans
nobody = Human()
Dave = Human()
Angela = Human()

# And lets set the names for a few
Dave.name = 'Dave'
Angela.name = 'Angela'

# and lets print these out
print(nobody.name) # None
print(Dave.name) # Dave
print(Angela.name) # Angela

# You can see how each instance of Human gets its own unique name, except for nobody who defaults to None as we specified.
# This lets us store data on an instance that is unique to the instance.
# However we can also have class variables. These are shared by all instances until an instance overrides it

class Canadian(Human):
	nationality = 'Canadian'

# lets create a few Canadians
Barry = Canadian()
Lucy = Canadian()
Tina = Canadian()

# I'll set their names agaiin
Barry.name = 'Barry'
Lucy.name = 'Lucy'
Tina.name = 'Tina'

# Lets print their names to show they are unique
print(Barry.name) # Barry
print(Lucy.name) # Lucy
print(Tina.name) # Tina

# And lets make sure they are Canadian because they share the same class variable
print(Barry.nationality) # Canadian
print(Lucy.nationality) # Canadian
print(Tina.nationality) # Canadian

# But now Tina wants to be German
Tina.nationality = 'German'
# Meanwhile all the Canadians now identify as North American
Canadian.nationality = 'North American'

# Lets check their nationalities
print(Barry.nationality) # North American
print(Lucy.nationality) # North American
print(Tina.nationality) # German

# You see that Tina is now German because that instance stores the value on itself.
# However the other instances of Canadian (Barry and Lucy) get it from the Class instead.
# This is one difference between a class variable and a Instance variable
# The other big difference is that class variables are defined when you import or load the python script, instance variables are defined when you create a new instance.

# But why didn't I use an __init__ in my Canadian class?
# This is because I inherit the __init__ from Human and I don't need to define anything more in the __init__ because I get it straight from Human instead
# But lets say I need to add on to what the __init__ in Human does. Lets try adding a gender. For this I'll add it as an instance variable even though a class variable would be fine

class Female(Human):
	def __init__(self):
		self.gender = 'F'

Linda = Female()

# Lets check if Linda has a name
print(hasattr(Linda, 'name')) # False

# This is because when we define our __init__ again, we override the one we got from Human.
# But lets say we want to use it as well

class Female(Human):
	def __init__(self):
		# We can use super to fetch the class we are inheriting from and run its init
		super(Female, self).__init__()
		# after which we define our custom behavior
		self.gender = 'F'

Linda = Female()

# And this time when we check if Linda has name set, it will be True
print(hasattr(Linda, 'name')) # True

