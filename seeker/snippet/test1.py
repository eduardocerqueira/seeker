#date: 2025-11-13T16:56:00Z
#url: https://api.github.com/gists/a73170b415158fb2010846e6015df6bd
#owner: https://api.github.com/users/mattsebastianh

load_file_in_context('script.py')
try:
  Circle
except NameError:
  fail_tests("Is there a class defined called `Circle`?")
try:
  circle1 = Circle(1)
  circle2 = Circle(4)
  circle3 = Circle(10)
except TypeError:
  fail_tests("Does `Circle` have a constructor that takes two parameters: `self` and `diameter`?")

if str(circle1) != "Circle with radius 1":
  fail_tests("Does `Circle` have a `__repr__` method that returns 'Circle with radius <X>' where <X> is the radius?")