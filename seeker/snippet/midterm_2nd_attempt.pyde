#date: 2021-10-25T17:09:34Z
#url: https://api.github.com/gists/b506b954c4917a5a7caa2d91afcbd086
#owner: https://api.github.com/users/sgodycki

"""
    Sasha Godycki
    CD Toolkit Python Fall 2021
    Midterm 10.26.21
"""
level = 1 
circleY = 300
circleX = 300
circleDirection = 1
position = 0
startTime = 0
rectY = 200
rectDirection =1
def setup():
    global img 
    size(576,720)
    img = loadImage("grass background.png")
    stroke(50,50,150)

    

def level1():
    image(img,0,0)
    global circleY, circleDirection, startTime, position
    fill(150,50,200)
    rect(position,200, 50,50)
    # move for a little while, then stop
    if millis() > startTime and millis() < startTime + 2000:
        position = position + 1

    # wait a little while, then reset the startTime variable,
    # so the above timing starts over:
    if millis() > startTime + 4000:
      startTime = millis()
    global rectY
    fill(0)
    rect(200,rectY,50,50)
    global rectDirection 
    rectY= rectY + rectDirection
    if rectY > width:
        rectDirection = -1
     
    fill(102,51,0)
    ellipse(300,circleY, 50,50) 
    circleY = circleY + circleDirection
    

    if circleY > width:
        circleDirection = -1
    if circleY < 0:
        level =2
    if keyPressed:
        if key == 'j':
            circleDirection = -1

    if key == 'l':
        circleDirection =1
    
     
def level2():
    
    global circleY, circleDirection, startTime, position
    image(img,0,0)
    fill(150,50,200)
    rect(position,200, 50,50)
    # move for a little while, then stop
    if millis() > startTime and millis() < startTime + 2000:
        position = position - 1

    # wait a little while, then reset the startTime variable,
    # so the above timing starts over:
    if millis() > startTime + 4000:
      startTime = millis()
    global rectY
    fill(250,150,50)
    rect(200,rectY,50,50)
    global rectDirection 
    rectY= rectY + rectDirection
    if rectY > width:
        rectDirection = -1
     
    fill(102,51,0)
    ellipse(300,circleY, 50,50) 
    circleY = circleY + circleDirection
    if circleY > width:
        circleDirection = -1
    if circleY < 0:
      circleDirection = 1
    if keyPressed:
        if key == 'j':
            circleDirection = -1

    if key == 'l':
        circleDirection =1
def draw():
    global level
    if level == 1:
        level1()
    if level ==2: 
        level2()
        
