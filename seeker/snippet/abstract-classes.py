#date: 2025-02-04T17:08:02Z
#url: https://api.github.com/gists/2873d5a9615400c089a4dad44116fe6a
#owner: https://api.github.com/users/docsallover

from abc import ABC, abstractmethod

class Shape(ABC): 
    @abstractmethod 
    def area(self): 
        pass 

class Circle(Shape): 
    # ... implementation of area() method ... 

class Rectangle(Shape): 
    # ... implementation of area() method ...