#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# circle.py

import color

from kandinsky import *
from gameobjects import *

class Circle(GameObject):
    def __init__(self, name, x, y, radius, backgroundColor = color.WHITE):
        GameObject.__init__(self, name, x, y)
        self.radius = radius
        self.backgroundColor = backgroundColor

    # radius

    def getRadius(self):
        return self.radius

    def setRadius(self, radius):
        self.radius = radius

    # draw ^-^

    # https://www.numworks.com/fr/ressources/python/activites/cercle/
    def draw(self):
        pass 


    
