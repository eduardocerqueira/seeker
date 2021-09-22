#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# sprite.py

from kandinsky import *

from gameobject import *

class Sprite(GameObject):

    def __init__(self, name: str, x: int, y: int, width: int, height: int, data):
        GameObject.__init__(self, name, x, y)
        self.width, self.height = width, height
        self.data = data

    # dimensions 

    def getDimensions(self):
        return (self.width, self.height)

    def setDimensions(self, width: int, height: int):
        self.width, self.height = width, height

    def setWidth(self, width: int):
        self.width = width

    def getWidth(self) -> int:
        return self.width

    def setHeight(self, height: int):
        self.height = height

    def getHeight(self) -> int:
        return self.height

    # data

    def setData(self, data):
        self.data = data

    def getData(self):
        return self.data

    # draw ^_^

    def draw(self):
        
        for y in range(self.height):
            for x in range(self.width):
                set_pixel(x + self.x, y + self.y, self.data[y][x])
