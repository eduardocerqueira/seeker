#date: 2023-04-18T16:49:26Z
#url: https://api.github.com/gists/40f084d90d5d767adbd54266a2075c45
#owner: https://api.github.com/users/Riwk

class Point:
    def __int__(self, x, y):
        self.x = x                      # a constructor is a function that gets call at the time of creating an object.
        self.y = y                                   # it is de__init__(self) it's short from of initialized

    @staticmethod
    def move():
        print("move")

    @staticmethod
    def draw():
        print("draw")


point = Point()
point.x = 22
point.y = 85
print(point.x)
print(point.y)
point.move()
point.draw()
