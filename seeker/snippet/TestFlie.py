#date: 2023-03-28T16:51:19Z
#url: https://api.github.com/gists/d1d134394baf27ca86dd45bd6fbc8243
#owner: https://api.github.com/users/ProjektPendelF

import pygame
from pygame.locals import *
from math import *
from datetime import datetime
#import matplotlib.pyplot as plt
import numpy as np
import pygame.gfxdraw

# Konstanten
g = 98.1
fps = 60
Zentrum = [400, 150]
fensterkoords = [800, 600]
now = datetime.now()
tstart = current_time = now.strftime("%H:%M:%S")
l = 250
ruhekoordinate = [400, Zentrum[1] + l]

# hier befinden sich alle Kombinationen aus x und y an denen das Pendel sich aufhalten kann
# (unterhalb des Zentrums und ohne Berücksichtigung der Geschwindigkeit in der ruhelage)
pendelkoordinaten = []
# array mit Schleife füllen
for x in range(ruhekoordinate[0] - l, ruhekoordinate[0] + l + 1):
    try:

        y = sqrt(pow(l, 2) - pow(Zentrum[0] - x, 2)) + Zentrum[1]
        pendelkoordinaten.append([x, y])
    except ValueError:
        print('Error')

# Zur Überprüfung des Arrays einfach die Kommentierung aufheben
'''
for i in range(0, len(pendelkoordinaten)):
    print(pendelkoordinaten[i])
'''

# FARBEN
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)

# init Frame
Frame = pygame.display
Frame.set_caption('Pendelvisualisierung')
Surface = Frame.set_mode(fensterkoords)
pygame.font.init()


# alle Werte, die direkt mit der Kugel zu tun haben
class Kugel():
    def __init__(self, radius=25, dichte=0.0025, color=YELLOW):
        self.shown = True
        self.radius = radius
        self.dichte = dichte
        self.gewicht = None
        self.setGewicht()
        self.color = color
        self.x, self.y = ruhekoordinate
        self.xGes = self.yGes = vGes = 0
        self.xBes = self.yBes = 0

    def zeichnen(self):
        pygame.draw.circle(Surface, self.color, (self.x, self.y), self.radius)

    def hide(self):
        self.shown = False

    def show(self):
        self.shown = True

    '''getter'''

    def getRadius(self):
        return self.radius

    def getDichte(self):
        return self.dichte

    def getGewicht(self):
        return self.gewicht

    def getxGes(self):
        return self.xGes

    def getyGes(self):
        return self.yGes

    def getvGes(self):
        return self.vGes

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getHoehe(self):
        return Zentrum[1] + l - self.getY()

    def getxBes(self):
        return self.xBes

    def getyBes(self):
        return self.yBes

    def getWkin(self):
        return 1 / 2 * self.getGewicht() * pow(self.getvGes(), 2)

    def getWpot(self):
        return self.getGewicht() * Physics.Fg

    '''setter'''

    def setRadius(self, radius):
        self.radius = radius
        self.setGewicht()

    def setDichte(self, dichte):
        self.dichte = dichte
        self.setGewicht()

    def setGewicht(self):
        self.gewicht = 4 / 3 * pi * pow(self.radius, 3) * self.dichte

    def setxGes(self, xGes):
        self.xGes = xGes

    def setyGes(self, yGes):
        self.yGes = yGes

    def setX(self, x):
        self.x = x
        self.setvGes()

    def setY(self, y):
        self.y = y
        self.setvGes()

    def setvGes(self):
        self.vGes = sqrt(pow(self.getyGes(), 2) + pow(self.getxGes(), 2))

    def setxBes(self, xGes):
        self.xBes = xGes

    def setyBes(self, yGes):
        self.yBes = yGes


class Physics():
    def __init__(self, Kugel):
        ### TODO mit Try versehen, solange die Variablen noch nicht gut definiert sind
        try:
            T = 2
            self.ymax = Zentrum[1] + l
            self.omega = 2 * pi / T  # Winkelv 2pi/T
            v = None  # omega * ymax * cos(omega * t)
            self.T = None  # pi / sqrt(l / g)  # Periodendauer T = 2pi/√(l/g)
            self.Fg = Kugel.getGewicht() * g  # m*a
            self.Fz = None  # Kugel.getGewicht() * pow(omega, 2)  # mw^2
            self.Wkin = None  # Kugel.getGewicht() * pow(Kugel.getvGes(), 2)
            self.Wpot = Kugel.getGewicht() * self.Fg
        except:
            pass

    # RECHNEN

    # Schwerkraft simulieren
    def grav(self, Kugel, g, fps):
        Kugel.setyBes(g)

    def bewegen(self, Kugel, fps):
        Kugel.setY(Kugel.getY() + 1 / fps * Kugel.getyGes())
        Kugel.setX(Kugel.getX() + 1 / fps * Kugel.getxGes())

    def aufspalten(self, groesse, winkel):
        # Winkel zwischen Vertikale und Hypotenuse

        ##### PYCHARM BENUTZT BOGENMAß

        # entspricht Ausschlag nach rechts
        x = sin(winkel) * groesse
        y = cos(winkel) * groesse


        if abs(x) < 0.000000001:
            x = 0
        if abs(y) < 0.000000001:
            y = 0

        return (x, y)

    # RENDER

    def Pfeil(self, color=RED, startpos=(0, 0), endpos=(0, 0), width=1):
        pygame.draw.line(Surface, color, startpos, endpos, width)

    def Pendelbogen(self, pendelkoordinaten):
        for i in range(0, len(pendelkoordinaten)):
            pygame.draw.circle(Surface, BLUE, (pendelkoordinaten[i][0], pendelkoordinaten[i][1]), 1)
            # print(pendelkoordinaten[i])
            # print(pendelkoordinaten[i][0], pendelkoordinaten[i][1])

    def PfeilGesamt(self, Kugel):
        self.PfeilBeschl(Kugel)
        self.PfeilGeschw(Kugel)

    def PfeilBeschl(self, Kugel, color=CYAN):

        # Beschleunigung der Kugel
        pygame.draw.line(Surface, color, (Kugel.getX(), Kugel.getY()), (Kugel.getX() + Kugel.getxBes(), Kugel.getY()),
                         width=2)
        pygame.draw.line(Surface, color, (Kugel.getX(), Kugel.getY()), (Kugel.getX(), Kugel.getY() + Kugel.getyBes()),
                         width=2)
        pygame.draw.line(Surface, color, (Kugel.getX(), Kugel.getY()),
                         (Kugel.getX() + Kugel.getxBes(), Kugel.getY() + Kugel.getyBes()), width=2)
        arrow_points = [(Kugel.getX() + Kugel.getxBes() + 1, Kugel.getY() + Kugel.getyBes()),
                        (Kugel.getX() + Kugel.getxBes() + 10, Kugel.getY() + Kugel.getyBes() - 10),
                        (Kugel.getX() + Kugel.getxBes(), Kugel.getY() + Kugel.getyBes() - 10),
                        ((Kugel.getX() + Kugel.getxBes() - 10, Kugel.getY() + Kugel.getyBes() - 10))]
        pygame.draw.polygon(Surface, color, arrow_points)

    def PfeilGeschw(self, Kugel, color=BLUE):
        # Geschwindigkeit der Kugel
        pygame.draw.line(Surface, color, (Kugel.getX(), Kugel.getY()), (Kugel.getX() + Kugel.getxGes(), Kugel.getY()),
                         width=2)
        pygame.draw.line(Surface, color, (Kugel.getX(), Kugel.getY()), (Kugel.getX(), Kugel.getY() + Kugel.getyGes()),
                         width=2)
        pygame.draw.line(Surface, color, (Kugel.getX(), Kugel.getY()),
                         (Kugel.getX() + Kugel.getxGes(), Kugel.getY() + Kugel.getyGes()), width=2)


# TODO Text weiterbearbeiten
class Text():
    def __init__(self, msg='Nothing here yet', color=RED, font=pygame.font.Font("C:/Windows/Fonts/Arial.ttf", 30),
                 size=30,
                 x=200, y=100):
        self.text = font.render(msg, 1, color)
        self.textpos = self.text.get_rect()
        self.textpos.centerx = Surface.get_rect().centerx


# init
Kugel = Kugel()
Clock = pygame.time.Clock()
Physics = Physics(Kugel)
index = 0

Kraefte = Text(f'Hier wirken gerade {round(g * Kugel.getGewicht(), 2)}Newton', WHITE,
               pygame.font.Font("C:/Windows/Fonts/Arial.ttf", 10), 0, 0, 0)

# main loop
mode = 'running'
while mode == 'running' or mode == 'stopped':
    if mode == 'exit':
        break
    for event in pygame.event.get():
        if event.type == QUIT:
            mode = 'exit'
            print('Quit!')
        if event.type == K_ESCAPE:
            pygame.quit()
            mode = 'exit'
            print('Pressed Esc!')
        if event.type == K_SPACE:
            print('Pressed Space!')
            if mode == 'running':
                mode = 'stopped'
            else:
                mode = 'running'

        # Zum Testen des Fadens
        if event.type == K_UP:
            Kugel.setyGes(Kugel.getyGes() + 1)
        if event.type == K_DOWN:
            Kugel.setyGes(Kugel.getyGes() - 1)

        if event.type == K_RIGHT:
            Kugel.setxGes(Kugel.getxGes() + 1)
        if event.type == K_LEFT:
            Kugel.setxGes(Kugel.getxGes() - 1)

    # aktualisiert das Bild nur, wenn der Nutzer dies möchte
    if mode == 'running':
        t = now.strftime('%H:%M:%S')

        Surface.fill(BLACK)
        Physics.grav(Kugel, g, fps)
        Physics.bewegen(Kugel, fps)
        Physics.PfeilGesamt(Kugel)
        Physics.Pfeil(WHITE, Zentrum, (Kugel.getX(), Kugel.getY()), width=1)

        Kugel.zeichnen()
        Surface.blit(Kraefte.text, Kraefte.textpos)
        Physics.PfeilGeschw(Kugel)

        Physics.Pendelbogen(
            pendelkoordinaten
        )

        Frame.update()
        Clock.tick(fps)

pygame.quit()