#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# sprite.py

'''
    TODO : 

    - add scale system 
    - add rotation system
'''

class GameObject:
    def __init__(self, name: str, x: int, y: int):
        
        self.name = name

        self.x, self.y = x, y
        self.enabled = True

    # name

    def getName(self) -> str:
        return self.name

    def setName(self, name: str):
        self.name = name

    # position

    def setPosition(self, x, y):
        self.x, self.y = x, y

    def getPosition(self):
        return (self.x, self.y)

    def setX(self, x: int):
        self.x = x
    
    def setY(self, y: int):
        self.y = y

    def getX(self) -> int:
        return self.x
    
    def getY(self) -> int:
        return self.y

    # visibility 

    def setActive(self, x: bool):
        self.enabled = x

    def isActive(self) -> bool:
        return self.enabled

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
