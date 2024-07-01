#date: 2024-07-01T17:11:58Z
#url: https://api.github.com/gists/c725bdc0d0b7876e41cf055234973042
#owner: https://api.github.com/users/ggorlen

from turtle import Turtle


class Scoreboard:
    def __init__(self, x=0, y=0):
        self.t = t = Turtle()
        self.score = -1
        t.color("white")
        t.penup()
        t.goto(x, y)
        t.hideturtle()
        self.update_scoreboard()

    def update_scoreboard(self):
        self.t.write(
            f"Score: {self.score}", align="center", font=("Arial", 15, "normal")
        )

    def increase_score(self):
        self.score += 1
        self.t.clear()
        self.update_scoreboard()

    def game_over(self):
        self.t.goto(0, 0)
        self.t.write("Game Over", align="center", font=("Arial", 25, "normal"))