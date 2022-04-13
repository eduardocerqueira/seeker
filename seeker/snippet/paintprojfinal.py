#date: 2022-04-13T17:00:54Z
#url: https://api.github.com/gists/432790c031a65e7a7f367455f6a72aaa
#owner: https://api.github.com/users/IsaacJ60

# importing modules
from math import *
from tkinter import *
from tkinter import filedialog
from pygame import *

# initialize pygame and mixer settings
mixer.pre_init(44100, -16, 2, 2048)
init()

# removing tkinter window when quit
root = Tk()
root.withdraw()

# setting window size
WIDTH, HEIGHT = 1200, 800
# creating surfaces
screen = display.set_mode((WIDTH, HEIGHT))

# displaying window icon
icon = image.load("assets/icon.png")
display.set_icon(icon)
# setting window caption
display.set_caption("AOT Paint")

# setting colour rgb values
RED = (255, 0, 0)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# creating tool names and getting file names
toolNames = ["stamp0", "stamp1", "stamp2", "stamp3", "stamp4", "stamp5", "pencil",
             "eraser", "line", "fill", "rect", "ellipse", "brush", "clear", "text"]
toolImg = ["assets/pencil.png", "assets/eraser.png", "assets/line.png",
           "assets/paintbucket.png", "assets/rect.png", "assets/circle.png",
           "assets/brush.png", "assets/clear.png", "assets/text.png"]

# creating stamp names and loading files with transparency
stampNames = ["stamp0", "stamp1", "stamp2", "stamp3", "stamp4"]
stamps = [image.load("assets/stamp0.png").convert_alpha(),
          image.load("assets/stamp1.png").convert_alpha(),
          image.load("assets/stamp2.png").convert_alpha(),
          image.load("assets/stamp3.png").convert_alpha(),
          image.load("assets/stamp4.png").convert_alpha()]

# creating rect list for tools and stamps for collisions and other interactions
toolRects = [(280, 93, 65, 65), (363, 93, 65, 65), (440, 93, 65, 65), (516, 93, 65, 65),
             (588, 93, 65, 65), (685, 93, 60, 60)]
stampRects = []

# loading and blitting background & other theme images
bg = image.load("assets/bg.jpg").convert()
screen.blit(bg, (0, 0))
logo = transform.scale(image.load("assets/logo.png").convert_alpha(), (244, 70))
screen.blit(logo, (10, 50))

# loading and blitting colour palette w/ colour preview
colourPicker = image.load("assets/colourpicker.jpg").convert()
paletteRect = colourPicker.get_rect(topleft=(897, 40))
screen.blit(colourPicker, (897, 40))
currColRect = Rect(840, 40, 55, 55)
currColRect2 = Rect(840, 40, 55, 55)

# loading and blitting text "color" w/ font "Minecraft.ttf"
fontSmall = font.Font("assets/Minecraft.ttf", 15)
fontMed = font.Font("assets/Minecraft.ttf", 17)
colourText = fontSmall.render("COLOR", True, WHITE)
screen.blit(colourText, (840, 100))

# creating & drawing canvas with menu
canvasRect = Rect(275, 195, 900, 585)
canvasOutlineRect = Rect(270, 190, 910, 595)
draw.rect(screen, WHITE, canvasRect)
draw.rect(screen, BLACK, canvasOutlineRect, 5)

menuRect = Rect(0, 0, 1200, 33)
draw.rect(screen, (70, 10, 10), menuRect)

# creating and drawing stamp background
stampRect = Rect(275, 90, 500, 70)
draw.rect(screen, BLACK, stampRect, 0)
stampRect2 = Rect(275, 90, 500, 70)
draw.rect(screen, WHITE, stampRect2, 2)

# creating stamp adder rect and img
stampSize = 80
isUserStamp = False
stampAdderImg = image.load("assets/addstamp.png").convert_alpha()
screen.blit(stampAdderImg, (785, 110))
addStampRect = Rect(785, 105, 40, 40)
colourText = fontSmall.render("NEW", True, WHITE)
screen.blit(colourText, (695, 113))
colourText = fontSmall.render("STAMP", True, WHITE)
screen.blit(colourText, (689, 128))

# creating slider rects
sliderRect = Rect(550, 40, 250, 30)
sliderRect2 = Rect(550, 40, 264, 30)
thicknessText = fontSmall.render("TOOL THICKNESS", True, WHITE)

# creating menu buttons for save\load\undo\redo\fill
saveRect = Rect(10, 0, 40, 30)
saveimg = image.load("assets/save.png").convert_alpha()
screen.blit(saveimg, (10, 3))
loadRect = Rect(60, 0, 40, 30)
loadimg = image.load("assets/load.png").convert_alpha()
screen.blit(loadimg, (60, 3))
undoRect = Rect(110, 0, 40, 30)
undoimg = image.load("assets/undo.png").convert_alpha()
screen.blit(undoimg, (110, 4))
redoRect = Rect(160, 0, 40, 30)
redoimg = image.load("assets/redo.png").convert_alpha()
screen.blit(redoimg, (160, 4))
fillRect = Rect(210, 0, 40, 30)
fillimg = image.load("assets/fill.png").convert_alpha()
fillimg = transform.scale(fillimg, (35, 35))
screen.blit(fillimg, (210, -5))
musicRect = Rect(260, 0, 40, 30)
musicimg = image.load("assets/music.png").convert_alpha()
musicimg = transform.scale(musicimg, (33, 33))
screen.blit(musicimg, (260, 0))
voldownRect = Rect(310, 0, 30, 30)
voldownimg = image.load("assets/volumedown.png").convert_alpha()
voldownimg = transform.scale(voldownimg, (33, 33))
screen.blit(voldownimg, (310, 0))
volupRect = Rect(350, 0, 40, 30)
volupimg = image.load("assets/volumeup.png").convert_alpha()
volupimg = transform.scale(volupimg, (33, 33))
screen.blit(volupimg, (350, 0))
pauseplayRect = Rect(497, 42, 40, 30)
pauseplayimg = image.load("assets/pauseplay.png").convert_alpha()
pauseplayimg = transform.scale(pauseplayimg, (25, 25))

# setting tool to "", none selected
tool = ""

# default colour black
colour = BLACK

# initialize startx and starty for line, rect, ellipse
startx, starty = 0, 0

# initialize oldmx and oldmy for prev mouse pos
oldmx, oldmy = 0, 0

# initialize fix mouse coords for draw tool
fixdrawx, fixdrawy = 0, 0

# tool thickness
thk = 5

# undo/redo variables - check if action performed and count amount
action = False
undoList = []
redoList = []

# initialize boolean for if draw should be filled
toolFill = False

# initialize variable for which stamp is chosen
currStamp = -1

# loading and blitting tools and stamps, basic UI elements
loadCount = 0
loadCount2 = 0

# initialize music text movement var
musicTextMove = 0
musicMoveLimit = 0
musicVol = 0.3
released = False
isPlaying = False
musicText = ""
# creating music player rects and blitting
musicPlayerRect = Rect(275, 40, 250, 30)
musicSurface = Surface((250, 30))

# create pos surface
posRect = Rect(840, 120, 55, 43)
posSurface = Surface((56, 45))

# text
userText = ""

for i in range(196, 361, 80):  # "i" represents y-coord of img
    for j in range(25, 186, 80):  # "j" represents x-coord of img
        draw.rect(screen, GREY, (j, i, 70, 70))  # produce imgs in rows
        toolRects.append((j, i, 70, 70))

for i in range(211, 380, 80):
    for j in range(40, 200, 79):
        toolPics = image.load(toolImg[loadCount]).convert_alpha()
        screen.blit(toolPics, (j, i))
        loadCount += 1

for j in range(280, 600, 75):
    stampbg = image.load("assets/stampbg.png")
    trnstampbg = stampbg.convert_alpha().copy()
    trnstampbg.fill((255, 255, 255, 70), None, BLEND_RGBA_MULT)
    screen.blit(trnstampbg, (j, 90))
    stampRects.append((j, 93, 65, 65))

for j in range(280, 600, 77):
    s = transform.scale(stamps[loadCount2], (65, 65))
    screen.blit(s, (j, 93))
    loadCount2 += 1

# drawing backgrounds
bg1img = image.load("assets/bg1.png")
bg2img = image.load("assets/bg2.png")
bg3img = image.load("assets/bg3.png")
bg4img = image.load("assets/bg4.png")
bg1preview = transform.scale(bg1img, (115, 75))
bg2preview = transform.scale(bg2img, (115, 75))
bg3preview = transform.scale(bg3img, (115, 75))
bg4preview = transform.scale(bg4img, (115, 75))
screen.blit(bg1preview, (20, 710))
screen.blit(bg2preview, (145, 710))
screen.blit(bg3preview, (20, 625))
screen.blit(bg4preview, (145, 625))
bg1Rect = Rect(20, 710, 115, 75)
bg2Rect = Rect(145, 710, 115, 75)
bg3Rect = Rect(20, 625, 115, 75)
bg4Rect = Rect(145, 625, 115, 75)
draw.rect(screen, BLACK, bg1Rect, 3)
draw.rect(screen, BLACK, bg2Rect, 3)
draw.rect(screen, BLACK, bg3Rect, 3)
draw.rect(screen, BLACK, bg4Rect, 3)


# function for thickness slider
def slider():
    global thk, action  # thickness and undo/redo need to be accessed
    # drawing slider rects, need 2 so the slider doesn't show outside of rect
    draw.rect(screen, BLACK, sliderRect, 0)
    draw.rect(screen, WHITE, sliderRect, 2)
    draw.rect(screen, BLACK, sliderRect2, 0)
    draw.rect(screen, WHITE, sliderRect2, 2)
    screen.blit(thicknessText, (615, 48))
    if sliderRect.collidepoint(mx, my) and mb[0]:  # check for user moving slider
        thk = mx - 550  # chancing thickness based on position of slider
        draw.rect(screen, RED, Rect(thk + 550, 40, 15, 30))  # updating new pos
    draw.rect(screen, RED, Rect(thk + 550, 40, 15, 30))  # drawing slider when not clicked


# function for picking colour
def palette():
    global colour  # need to access colour var
    draw.rect(screen, WHITE, paletteRect, 2)  # drawing colour rect
    if paletteRect.collidepoint(mx, my) and mb[0]:  # checking for click on colours
        colour = screen.get_at((mx, my))  # getting colour at click pos
    # drawing colour preview
    draw.rect(screen, colour, currColRect, 0)
    draw.rect(screen, WHITE, currColRect2, 2)


# function for pencil tool
def pencil(thickness):
    global action  # access undo/redo status
    draw.line(screen, colour, (oldmx, oldmy), (mx, my), thickness)  # draw from prev mouse pos to current
    draw.circle(screen, colour, (mx, my), thickness)  # draw circle to make line look nicer
    action = True  # performed an action, set action to True


def brush(thickness):
    global action  # access undo/redo status
    draw.line(screen, colour, (oldmx, oldmy), (mx, my), thickness)
    draw.circle(screen, colour, (mx, my), thickness)
    action = True


# function for eraser tool
def eraser(thickness):
    global action  # access undo/redo status
    # drawing white line and circle to "erase" contents of screen
    draw.line(screen, WHITE, (oldmx, oldmy), (mx, my), thickness)
    draw.circle(screen, WHITE, (mx, my), thickness // 2)
    action = True  # action performed


# function for line tool
def line(sx, sy, thickness):
    global action  # access undo/redo status
    screen.blit(screenCap, (275, 195))  # blitting screen capture on canvas
    draw.line(screen, colour, (sx, sy), (mx, my), thickness)  # drawing line from mb down pos to current
    action = True  # action performed


# function for fill tool
def fill():
    global action  # access undo/redo status
    screen.subsurface(canvasRect).fill(colour)  # fill canvas with selected colour
    action = True  # action performed


# function for rect tool
def rect(sx, sy, thickness, isFilled):
    global action  # access undo/redo status
    rectKeys = key.get_pressed()
    screen.blit(screenCap, (275, 195))  # blitting screen capture on canvas
    if rectKeys[K_LSHIFT]:
        rectRect = Rect(sx, sy, (mx - sx), (mx - sx))  # blitting rect at new mouse pos
    else:
        rectRect = Rect(sx, sy, (mx - sx), (my - sy))  # blitting rect at new mouse pos
    rectRect.normalize()  # normalize rect to get rid of negative values
    draw.rect(screen, colour, rectRect, 0 if isFilled else thickness)  # blitting rect
    action = True  # action performed


# function for ellipse tool
def ellipse(sx, sy, isFilled, thickness):
    global action  # access undo/redo status
    ellipseKey = key.get_pressed()
    screen.blit(screenCap, (275, 195))  # blitting screen capture on canvas
    if ellipseKey[K_LSHIFT]:
        circleRect = Rect(sx, sy, mx - sx, mx - sx)  # creating rect for ellipse tool
    else:
        circleRect = Rect(sx, sy, mx - sx, my - sy)  # creating rect for ellipse tool
    circleRect.normalize()  # normalizing rect
    draw.ellipse(screen, colour, circleRect, 0 if isFilled else thickness)  # draw ellipse
    action = True  # action performed


# function for saving canvas
def saveTool():
    fileName = filedialog.asksaveasfilename(defaultextension=".png")
    if not fileName:
        return 0
    else:
        image.save(screen.subsurface(canvasRect), fileName)


# function for loading image to canvas
def loadTool():
    global action
    fileName = filedialog.askopenfilename()
    if fileName:
        loadedimg = image.load(fileName)
        if loadedimg.get_width() > 900:
            newHeight = loadedimg.get_height() * (900 / loadedimg.get_width())
            loadedimg = transform.scale(loadedimg, (900, newHeight))
        if loadedimg.get_height() > 585:
            newWidth = (loadedimg.get_width() * (900 / loadedimg.get_height()))
            loadedimg = transform.scale(loadedimg, (newWidth, 585))
        screen.blit(loadedimg, (275, 195))
        action = True


def createStamp(num):
    global action, currStamp, stampSize
    screen.blit(screenCap, (275, 195))
    screen.set_clip(canvasRect)
    if currStamp == 5:
        scaledUserStamp = transform.scale(userStamp, (stampSize, stampSize))
        screen.blit(scaledUserStamp, (mx - 40, my - 40))
        action = True
    else:
        scaledStamp = transform.scale(stamps[num], (stampSize, stampSize))
        screen.blit(scaledStamp, (mx - 40, my - 40))
        action = True


def musicPlayer():
    global musicTextMove, musicText, musicMoveLimit, released, isPlaying, musicVol
    screen.blit(musicSurface, (275, 40))
    screen.blit(pauseplayimg, (497, 42))
    draw.rect(screen, WHITE, musicPlayerRect, 2)
    musicSurface.blit(musicCap, (0, 0))
    if musicRect.collidepoint(mx, my) and mb[0] and released:
        musicName = filedialog.askopenfilename()
        released = False
        if musicName:
            isPlaying = True
            mixer.music.load(musicName)
            mixer.music.set_volume(musicVol)
            mixer.music.play()
            fileCutoff = musicName.rindex("/")
            musicName = musicName[fileCutoff+1:len(musicName)-4]
            musicMoveLimit = int(len(musicName) * 8.7 - 250)
            musicText = fontSmall.render(musicName, True, WHITE)
    elif musicText != "":
        # takes ~35 characters to reach end of rect
        musicSurface.blit(musicText, (15-musicTextMove, 7))
        musicTextMove += 0.04
        if musicTextMove > musicMoveLimit:
            musicTextMove = 0


def displayCoords():
    screen.blit(posSurface, (840, 120))
    draw.rect(screen, WHITE, posRect, 1)
    posxText = fontSmall.render("X " + str(mx), True, WHITE)
    posyText = fontSmall.render("Y " + str(my), True, WHITE)
    screen.blit(posxText, (844, 124))
    screen.blit(posyText, (844, 144))


# text function
def typeText(text):
    global action
    screen.blit(screenCap, (275, 195))
    screen.set_clip(canvasRect)
    if text != "":
        screen.set_clip(canvasRect)
        uText = fontMed.render(text, True, colour)
        screen.blit(uText, (mx + 10, my))


# distance between prev coords and current coords (fix for pencil tool)
def distance(prevX, prevY, currX, currY):  # take old and new pos
    return sqrt((prevX - currX) ** 2 + (prevY - currY) ** 2)  # return dist between pos


screenCap = screen.subsurface(canvasRect).copy()
musicCap = musicSurface.copy()

# screencapture of blank screen for undo/redo
actionCap = screen.subsurface(canvasRect).copy()
undoList.append(actionCap)

running = True
while running:
    mx, my = mouse.get_pos()  # getting the current mx and my
    displayCoords()
    mb = mouse.get_pressed()
    for evt in event.get():
        if evt.type == QUIT:
            running = False

        if evt.type == KEYDOWN:
            if evt.key == K_BACKSPACE:
                userText = userText[:-1]
            elif evt.key == K_RETURN:
                userText = ""
                tool = ""
            elif evt.key == K_ESCAPE:
                userText = ""
            elif evt.key == K_TAB:
                userText += "   "
            else:
                userText += evt.unicode

        if evt.type == MOUSEBUTTONDOWN:
            startx, starty = evt.pos

            for i in range(5):
                if Rect(stampRects[i]).collidepoint(mx, my):
                    currStamp = i
            if Rect(685, 95, 60, 60).collidepoint(mx, my) and isUserStamp:
                currStamp = 5

        if evt.type == MOUSEBUTTONUP:

            released = True

            if bg1Rect.collidepoint(mx, my):
                screen.blit(bg1img, (275, 195))
            if bg2Rect.collidepoint(mx, my):
                screen.blit(bg2img, (275, 195))
            if bg3Rect.collidepoint(mx, my):
                screen.blit(bg3img, (275, 195))
            if bg4Rect.collidepoint(mx, my):
                screen.blit(bg4img, (275, 195))

            if pauseplayRect.collidepoint(mx, my) and musicText != "":
                if not isPlaying:
                    mixer.music.unpause()
                    isPlaying = True
                else:
                    mixer.music.pause()
                    isPlaying = False

            if volupRect.collidepoint(mx, my) and (musicVol <= 10):
                musicVol += 0.1
                mixer.music.set_volume(musicVol)

            if voldownRect.collidepoint(mx, my) and (musicVol >= 0):
                musicVol -= 0.1
                mixer.music.set_volume(musicVol)

            if addStampRect.collidepoint(mx, my):
                stampName = filedialog.askopenfilename()
                if stampName:
                    isUserStamp = True
                    userStamp = image.load(stampName)
                    userStampIcon = transform.scale(userStamp, (60, 60))
                    draw.rect(screen, BLACK, Rect(680, 95, 60, 60))
                    screen.blit(userStampIcon, (680, 95))

            if saveRect.collidepoint(mx, my):
                saveTool()

            if loadRect.collidepoint(mx, my):
                loadTool()

            if fillRect.collidepoint(mx, my):
                if toolFill:
                    toolFill = False
                else:
                    toolFill = True

            if undoRect.collidepoint(mx, my):
                if len(undoList) > 1:
                    undoCap = undoList.pop()
                    redoList.append(undoCap)
                    screen.blit(undoList[-1], (275, 195))
                else:
                    screen.subsurface(canvasRect).fill(WHITE)
                action = False
            if redoRect.collidepoint(mx, my):
                if len(redoList) > 0:
                    redoCap = redoList.pop()
                    undoList.append(redoCap)
                    screen.blit(undoList[-1], (275, 195))
            if action and (tool != ""):
                actionCap = screen.subsurface(canvasRect).copy()
                undoList.append(actionCap)
                action = False

            screenCap = screen.subsurface(canvasRect).copy()

        rules = [canvasRect.collidepoint(mx, my) == False,
                 paletteRect.collidepoint(mx, my) == False,
                 sliderRect.collidepoint(mx, my) == False,
                 sliderRect2.collidepoint(mx, my) == False,
                 undoRect.collidepoint(mx, my) == False,
                 redoRect.collidepoint(mx, my) == False,
                 fillRect.collidepoint(mx, my) == False,
                 volupRect.collidepoint(mx, my) == False,
                 voldownRect.collidepoint(mx, my) == False]

        for i in range(len(toolNames)):
            if Rect(toolRects[i]).collidepoint(mx, my) or tool == toolNames[i]:
                if "stamp" not in toolNames[i]:
                    draw.rect(screen, RED, toolRects[i], 2)
                if evt.type == MOUSEBUTTONUP and all(rules):
                    if tool == toolNames[i]:
                        tool = ""
                    else:
                        tool = toolNames[i]
            elif "stamp" not in toolNames[i]:
                draw.rect(screen, GREEN, toolRects[i], 2)

    slider()
    palette()
    musicPlayer()

    # creating stamps
    if "stamp" in tool:
        createStamp(currStamp)

    # use tools
    if canvasRect.collidepoint(mx, my):
        screen.set_clip(canvasRect)
        if mb[0]:
            if tool == "pencil":
                pencil(1)
                dx = mx - oldmx
                dy = my - oldmy
                apartDist = int(distance(oldmx, oldmy, mx, my))

                for i in range(1, apartDist):
                    fixdrawx = dx * i / apartDist + oldmx
                    fixdrawy = dy * i / apartDist + oldmy
                    draw.circle(screen, colour, (int(fixdrawx), int(fixdrawy)), 1)
            if tool == "brush":
                brush(thk)
                dx = mx - oldmx
                dy = my - oldmy
                apartDist = int(distance(oldmx, oldmy, mx, my))

                for i in range(1, apartDist):
                    fixdrawx = dx * i / apartDist + oldmx
                    fixdrawy = dy * i / apartDist + oldmy
                    draw.circle(screen, colour, (int(fixdrawx), int(fixdrawy)), thk)
            if tool == "eraser":
                eraser(thk)
            if tool == "line":
                line(startx, starty, thk)
            if tool == "fill":
                fill()
            if tool == "rect":
                rect(startx, starty, thk, toolFill)
            if tool == "ellipse":
                ellipse(startx, starty, toolFill, thk)
        if tool == "text":
            typeText(userText)

    if released and tool == "clear":
        screen.subsurface(canvasRect).fill(WHITE)
        released = False

    oldmx, oldmy = mx, my
    display.flip()
    screen.set_clip(None)
quit()
