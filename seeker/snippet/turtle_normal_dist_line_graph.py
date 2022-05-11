#date: 2022-05-11T17:22:27Z
#url: https://api.github.com/gists/722939e331c7ba1cd17be5db3b777367
#owner: https://api.github.com/users/khurchla

# import the turtle library
import turtle

# add a graph paper image from file as a background
turtle.bgpic("Probability_Graph_Paper_Template-landscape.png")

# change the shape back to the cute little turtle
turtle.shape("turtle")
# give it a good name (optional but motivating)
squirtle = turtle
# color it green
squirtle.color("aqua")
# set the line thickness
squirtle.pensize(3)

# by default turtles starts at the center (0,0) of the screen
# face it southwest and go to the bottom left of the screen without making a mark
squirtle.setheading(225)
squirtle.forward(350)

# set the orientation of the turtle to face 45 degrees towards the northeast now
squirtle.setheading(45)

# now move turtle forward
squirtle.forward(550)

# add a text call-out, but in a different color
squirtle.color('red')
squirtle.write("Hey, does this look normal?", align="center", font=("courier new", 25, "bold"))
# change squirtle's color back to aqua after writing
squirtle.color("aqua")

# start a screen event loop - calling Tkinter’s mainloop function
# aka squirtle.mainloop() (note tkinter methods are working in the background)
squirtle.done()