#date: 2024-07-01T17:11:58Z
#url: https://api.github.com/gists/c725bdc0d0b7876e41cf055234973042
#owner: https://api.github.com/users/ggorlen

from turtle import Turtle


class Snake:
    def __init__(self, move_dist):
        self.move_dist = move_dist
        self.segments = []
        positions = [(0, 0), (-move_dist, 0), (-move_dist * 2, 0)]
        self.create_snake(positions)
        self.head = self.segments[0]

    def create_snake(self, positions):
        for snake_body in positions:
            self.add_segment()

    def add_segment(self):
        segment = Turtle("square")
        segment.penup()
        segment.hideturtle()
        segment.color("white")
        segment.setpos(x=-self.move_dist, y=0)
        self.segments.append(segment)

    def extend(self):
        self.add_segment()

    def move(self):
        for seg_num in range(len(self.segments) - 1, 0, -1):
            # coord of previous segment
            new_x = self.segments[seg_num - 1].xcor()
            new_y = self.segments[seg_num - 1].ycor()

            # previous segment moves to coord of next segment
            self.segments[seg_num].goto(new_x, new_y)
            self.segments[seg_num].showturtle()

        self.head.forward(self.move_dist)

    def turn(self, direction):
        if (
            direction - 180 != self.head.heading()
            and direction + 180 != self.head.heading()
        ):
            self.head.setheading(direction)

    def collides_with_food(self, food):
        return self.head.distance(food.pos()) < self.move_dist

    def collides_with_tail(self):
        for segment in self.segments[1:]:
            if self.head.distance(segment) < self.move_dist // 2:
                return True

        return False

    def collides_with_wall(self, w, h):
        return (
            self.head.xcor() > w / 2 + self.move_dist / 2 # right side
            or self.head.xcor() < -w / 2 + self.move_dist / 2 # left side
            or self.head.ycor() > h / 2 - self.move_dist / 2 # top
            or self.head.ycor() < -h / 2 + self.move_dist # bottom
        )