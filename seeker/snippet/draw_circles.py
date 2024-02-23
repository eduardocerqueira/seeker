#date: 2024-02-23T17:07:48Z
#url: https://api.github.com/gists/5a3daf1aa9ca6cd5226e07137d8983bb
#owner: https://api.github.com/users/mr-gideon

# Part 1: Draw a circle
import turtle

tina = turtle.Turtle()

tina.color('blue')
tina.shape('turtle')

tina.speed(10)
tina.pensize(4)
tina.circle(60)

# Part 2:
# Position a circle
# Note, the penup and pendown commands
# Optional: Show how to fill a circle

tina.penup()
tina.goto(100,100)
tina.pendown()
# tina.color("red","black")
# tina.begin_fill()
tina.circle(40)
# tina.end_fill()