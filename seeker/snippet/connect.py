#date: 2022-03-02T17:11:45Z
#url: https://api.github.com/gists/231e4d5a166637eaa84d868ff0c2edb4
#owner: https://api.github.com/users/albionbrown

#!/bin/python3
import turtle
import time

circles = []
startx = -150
starty = 150

circles_per_row = 7
distance_between_circles = 40
circle_counter = 0

for i in range(1, 43):
  new_circle = turtle.Turtle()
  new_circle.shape('circle')
  new_circle.speed('fastest')
  new_circle.penup()
  
  if circle_counter >= circles_per_row:
    starty = starty - 40
    startx = -150
    circle_counter = 0
    
  new_circle.setx(startx)
  new_circle.sety(starty)
  
  startx = startx + distance_between_circles
  
  circles.append(new_circle)
  circle_counter = circle_counter + 1

