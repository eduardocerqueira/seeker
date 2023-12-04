#date: 2023-12-04T17:02:35Z
#url: https://api.github.com/gists/2aee350f5acd7a0988a918b0b8296f99
#owner: https://api.github.com/users/rafea25

class Entity:
    def __init__(self, max_size, x, y):
        self.x = x
        self.y = y
        self.max_size = max_size
    def __str__(self) -> str:
        return f"({self.x}), ({self.y})"
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    def action(self, choice):
        if choice == 0:
            self.move(1,0)
            return (1,0)
        if choice == 1:
            self.move(-1,0)
            return (-1,0)
        if choice == 2:
            self.move(0,1)
            return (0,1)
        if choice == 3:
            self.move(0,-1)
            return (0,-1)
    def theoretical_action(self,choice):
      #this doesn't actually move the entity it just returns what the new location would be if we take the action
      #this enables us to use this potential location such that if we eat an apple we add a new snake body to the top
      #of the list if we havent eaten the snake we move the tail to this theoretical location
        if choice == 0:
            return (self.x + 1, self.y)
        if choice == 1:
            return (self.x - 1, self.y)
        if choice == 2:
            return (self.x, self.y + 1)
        if choice == 3:
            return (self.x, self.y - 1)
    def move(self, x, y):
        self.x += x
        self.y += y
    def get_coord(self):
        return (self.x, self.y)
    def is_out_of_bounds(self):
        return not (self.x >= 0 and self.x < self.max_size and self.y >= 0 and self.y < self.max_size)
    

