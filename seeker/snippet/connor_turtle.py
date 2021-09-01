#date: 2021-09-01T13:14:16Z
#url: https://api.github.com/gists/1d4baef1a79cb8e40c9f8b9a8ca03873
#owner: https://api.github.com/users/crichmond-clark

import turtle
t = turtle.Turtle()
t.speed(5) # 1:slowest, 3:slow, 5:normal, 10:fast, 0:fastest
t.pencolor('purple')
t.pensize(5)
LINE_LENGTH = 40

def space(spacing):
  t.penup()
  t.sety(90)
  t.setx(t.xcor() + spacing)
  t.pendown()

def draw_line(angle, length):
  t.setheading(angle)
  t.forward(length)

def draw_c():
	t.circle(-20, -180)

def draw_o():
  t.setheading(-180)
  t.circle(20)

def draw_n():
  draw_line(-90, LINE_LENGTH)
  t.penup()
  draw_line(90, LINE_LENGTH)
  t.pendown()
  draw_line(-65, LINE_LENGTH + 5)
  draw_line(90, LINE_LENGTH)

def draw_r():
  draw_line(-90, LINE_LENGTH)
  t.penup()
  draw_line(90, LINE_LENGTH)
  t.pendown()
  t.setheading(0)
  t.circle(-12, 180)
  draw_line(-60, 20)

alphabet = {
  "c": draw_c, "o": draw_o, "n": draw_n, "r": draw_r
}

def draw_name(name):
  name = name.lower()
  t.penup()
  t.goto(-90,90)
  t.pendown()
  for letter in name:
    alphabet[letter]()
    if letter == "n":
      space(20)
    else:
    	space(30)

draw_name("connor")