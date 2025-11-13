#date: 2025-11-13T16:56:00Z
#url: https://api.github.com/gists/a73170b415158fb2010846e6015df6bd
#owner: https://api.github.com/users/mattsebastianh

class Circle:
  pi = 3.14

  def __init__(self, diameter):
    self.radius = diameter / 2

  def __repr__(self):
    return "Circle with radius {}".format(self.radius)
  
  def area(self):
    return self.pi * self.radius ** 2
  
  def circumference(self):
    return 2 * self.pi * self.radius
  
medium_pizza = Circle(12)
teaching_table = Circle(36)
round_room = Circle(11460)

print(medium_pizza)
print(teaching_table)
print(round_room)