#date: 2021-12-08T17:11:58Z
#url: https://api.github.com/gists/7c15200d17388a2dea2a2ff7f62af05a
#owner: https://api.github.com/users/fumanchez

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return f"Rectangle {self.width}x{self.height}"

    def area(self): return self.width * self.height

    def is_square(self): return self.width == self.height

    def is_vertical(self): return self.width < self.height

    def is_horizontal(self): return self.width > self.height

    def max_inner_square(self):
        if self.is_square(): return self

        if self.is_horizontal():
            residual_rectangle = Rectangle(self.width - self.height, self.height)
        else:
            residual_rectangle = Rectangle(self.width, self.height - self.width)

        return residual_rectangle.max_inner_square()


rectangle = Rectangle(1680, 640)
max_inner_square = rectangle.max_inner_square()

print(f"{max_inner_square} is max inner square for {rectangle}")
