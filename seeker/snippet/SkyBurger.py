#date: 2024-02-08T17:03:42Z
#url: https://api.github.com/gists/ad461b20b49f3d3a6c1ca54d8223de0a
#owner: https://api.github.com/users/kushedow

class Burger:

    def __init__(self, bun=None, patties=None, greens=None, cheese=None, sauces=None):

        self.bun: str = bun 
        self.greens: list = greens 
        self.cheese: list = cheese 
        self.sauces: list = sauces
        self.patties: list = patties

    def __repr__(self):
        return f"Burger({self.bun},{self.patties},{self.greens},{self.cheese},{self.sauces})"
