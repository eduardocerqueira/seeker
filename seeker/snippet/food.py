#date: 2024-07-01T17:11:58Z
#url: https://api.github.com/gists/c725bdc0d0b7876e41cf055234973042
#owner: https://api.github.com/users/ggorlen

from random import randint
from turtle import Turtle


class Food:
    def __init__(self, move_dist):
        self.move_dist = move_dist
        self.t = t = Turtle()
        t.shape("circle")
        t.penup()
        t.shapesize(0.6)
        t.color("blue")

    def change_pos(self, w, h):
        self.t.goto(
            randint(-w / 2 + self.move_dist, w / 2 - self.move_dist),
            randint(-h / 2 + self.move_dist, h / 2 - self.move_dist),
        )

    def pos(self):
        return self.t.pos()