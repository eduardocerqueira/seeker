#date: 2025-06-17T17:10:38Z
#url: https://api.github.com/gists/66b3263b248231f065a0ded3eb162ba8
#owner: https://api.github.com/users/Jeihunn

# Flyweight: stores shared state
class TreeType:
    def __init__(self, species, color):
        self._species = species
        self._color = color

    def render(self, x, y):
        print(f"{self._species}({self._color}) at ({x},{y})")

# Factory: manages flyweight pool
class TreeFactory:
    _flyweights = {}

    @classmethod
    def get_type(cls, species, color):
        key = (species, color)
        if key not in cls._flyweights:
            print(f"Creating flyweight: {key}")
            cls._flyweights[key] = TreeType(species, color)
        return cls._flyweights[key]

# Context: holds unique state
class Tree:
    def __init__(self, x, y, tree_type):
        self.x, self.y = x, y  # Extrinsic state
        self.type = tree_type  # Flyweight reference

    def render(self):
        self.type.render(self.x, self.y)

# Client code
if __name__ == "__main__":
    trees = [
        Tree(10, 20, TreeFactory.get_type("Oak", "Green")),
        Tree(15, 30, TreeFactory.get_type("Oak", "Green")),  # Reuses
        Tree(50, 80, TreeFactory.get_type("Pine", "Dark")),
        Tree(25, 40, TreeFactory.get_type("Oak", "Green")),  # Reuses
        Tree(90, 39, TreeFactory.get_type("Pine", "Dark")),  # Reuses
    ]

    for tree in trees:
        tree.render()

    print(f"Flyweights created: {len(TreeFactory._flyweights)}")

    # Output:
    # Creating flyweight: ('Oak', 'Green')
    # Creating flyweight: ('Pine', 'Dark')
    # Oak(Green) at (10,20)
    # Oak(Green) at (15,30)
    # Pine(Dark) at (50,80)
    # Oak(Green) at (25,40)
    # Pine(Dark) at (90,39)
    # Flyweights created: 2