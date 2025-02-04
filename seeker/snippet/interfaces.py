#date: 2025-02-04T17:08:57Z
#url: https://api.github.com/gists/d8ffe09bc24668846511458d5e8aae59
#owner: https://api.github.com/users/docsallover

from abc import ABC, abstractmethod

class Drawable(ABC): 
    @abstractmethod 
    def draw(self): 
        pass 

class Circle(Drawable): 
    # ... implementation of draw() method ... 

class Rectangle(Drawable): 
    # ... implementation of draw() method ...