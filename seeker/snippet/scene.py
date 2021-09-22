#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# scene.py

import screen
import color

class Scene:
    def __init__(self, backgroundColor = color.WHITE):
        self.gameobjects = []
        self.backgroundColor = backgroundColor

    # gameobjects list

    def addGameObject(self, obj):
        self.gameobjects.append(obj)

    def removeGameObject(self, obj):
        self.gameobjects.remove(obj)

    def countGameObjects(self):
        return len(self.gameobjects)

    def clear(self):
        self.gameobjects.clear()

    def getGameObjectsByName(self, name):

        o = []
        for gameobject in self.gameobjects:
            if name == gameobject.name:
                o.append(gameobject)

        return o


    def getGameObjectsByType(self, t):

        o = []
        for gameobject in self.gameobjects:
            if type(gameobject) == t:
                o.append(gameobject)
                
        return o

    # backgroundColor

    def setBackgroundColor(self, backgroundColor):
        self.backgroundColor = backgroundColor

    def getBackgroundColor(self):
        return self.backgroundColor

    # draw

    def draw(self):
        screen.fill(self.backgroundColor)

        for gameobject in self.gameobjects:
            if gameobject.isActive():
                gameobject.draw()            # draws the object if it's visible
