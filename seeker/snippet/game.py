#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# game.py

class Game:
    def __init__(self):
        self.currentScene = None

    def start(self):
        pass

    def update(self):
        self.draw()

    def draw(self):

        if self.currentScene == None:
            return

        self.currentScene.draw()

    # scene management

    def setCurrentScene(self, scene):
        self.currentScene = scene

    def getCurrentScene(self):
        return self.currentScene
