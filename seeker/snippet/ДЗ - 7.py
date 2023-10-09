#date: 2023-10-09T17:01:12Z
#url: https://api.github.com/gists/590c7ac938ed5d79cb1e0c2896487a7f
#owner: https://api.github.com/users/Mnemo2005

#----------------------задача 4 ----------------------
#  напишите программу для черепахи, чтобы она рисовала вот так 
#   (кол-во сторон произвольное)

from turtle import *
pensize(2)
speed(1)
n = 10
for i in range(1):
    color('blue')
    forward(50)
    left(90)
    color('red')
    forward(50)
    left(90)
    color('green')
    forward(50+n)
    left(90)
    color('blue')
    forward(50+n)
    left(90)
    color('red')
    forward(50+n*2)
    left(90)
    color('green')
    forward(50 + n*2)



# ----------------задача 5-------------------
#  напишите программу для черепахи, чтобы она рисовала вот так
#    (кол-во сторон произвольное)
from turtle import *
pensize(2)
speed(1)
for i in range(2):
    color('blue')
    forward(50)
    left(90)
    color('blue')
    forward(50)
    right(90)
    color('blue')
    forward(50)
    right(90)
    color('blue')
    forward(50)
    left(90)
    color('green')
    forward(50)
    left(90)
    color('green')
    forward(50)
    right(90)
    color('green')
    forward(50)
    right(90)
    color('green')
    forward(50)
    left(90)
    color('red')
    forward(50)
    left(90)
    color('red')
    forward(50)
    right(90)
    color('red')
    forward(50)
    right(90)
    color('red')
    forward(50)
    left(90)







