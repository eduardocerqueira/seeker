#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# rectangle.py

from gameobject import *

class Rectangle(GameObject):

    def __init__(self, name: str, x: int, y: int, width: int, height: int, backgroundColor = color.WHITE):
        GameObject.__init__(self, name, x, y)
        self.width, self.height = width, height
        self.backgroundColor = backgroundColor

    # dimensions 

    def getDimensions(self):
        return (self.width, self.height)

    def setDimensions(self, width, height):
        self.width, self.height = width, height

    def setWidth(self, width: int):
        self.width = width

    def getWidth(self) -> int:
        return self.width

    def setHeight(self, height: int):
        self.height = height

    def getHeight(self) -> int:
        return self.height

    # backgroundColor

    def getBackgroundColor(self):
        return self.backgroundColor

    def setBackgroundColor(self, backgroundColor):
        self.backgroundColor = backgroundColor

    # draw

    def draw(self):
        fill_rect(self.x, self.y, self.width, self.height, self.backgroundColor)
