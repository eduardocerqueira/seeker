#date: 2021-11-05T16:49:17Z
#url: https://api.github.com/gists/19895add83c5cd762d18e6721a2473bb
#owner: https://api.github.com/users/treyhunner

class Rectangle:

    def __init__(self, width, height):
        self.width, self.height = width, height

    def __repr__(self):
        return f"Rectangle({self.width}, {self.height})"
 
    @property
    def area(self):
        return self.width * self.height

    @area.setter
    def area(self, area):
        self.width *= area/self.area