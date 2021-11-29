#date: 2021-11-29T17:04:57Z
#url: https://api.github.com/gists/3f211889ca2a6d453b82554751be09e8
#owner: https://api.github.com/users/pigmonchu

# Jerarquía de clases para representar personajes de Star Wars
class StarWarsCharacter:
    def __init__(self, name):
        self.name = name
        
class Jedi (StarWarsCharacter):
    def __init__(self, name, midichlorians):
        super().__init__(name)
        self.midichlorians = midichlorians

