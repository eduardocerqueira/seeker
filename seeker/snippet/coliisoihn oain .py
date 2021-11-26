#date: 2021-11-26T17:07:33Z
#url: https://api.github.com/gists/fcf9c408abff0a8630a8d0e1a972cd80
#owner: https://api.github.com/users/ValkyrieXD

import pygame
import math
import random
from pygame import *
from random import *


# initialize modules, wait.. import is USELESS?!?!?!
pygame.init()

# create screen
screen = pygame.display.set_mode((800,600))

#Title and icon
pygame.display.set_caption("Cave Explore")
icon = pygame.image.load('sources/textures/icon.png')
pygame.display.set_icon(icon)

#player wooo
playerImg = pygame.image.load("sources/textures/bop.png")
playerX = 370
playerY = 480
playerX_change = 0
playerY_change = 0
rect1 = playerImg.get_rect()

def player(x,y):
    screen.blit(playerImg, (rect1.x, rect1.y))

#Enemy
enemyImg = pygame.image.load("sources/textures/redboi.png")
enemyY = randint(0, 800)
enemyX = randint(50, 550)
enemyX_change = 0.3
enemyY_change = 0.3


#gun
gunImg = pygame.image.load("sources/textures/gun.png")
gunX = 418
gunY = 500
gunX_change = 0
gunY_change = 0

def gun():
    screen.blit(gunImg, (gunX, gunY))

#bullet
bulletImg = pygame.image.load("sources/textures/bullet.png")
bulletX= gunX
bulletX_change = 0
bulletY_change = 0.2
bulletState ="ready"

#[placeholder]
spawnrect = pygame.Rect(playerX,playerY,100,100)


def fire_bullet(x,y):
    global bulletState
    bulletState = "fire"
    bulletImg = pygame.image.load("sources/textures/bullet.png")
    screen.blit(bulletImg, (x + gunX ,y + playerY))

def enemy():
    screen.blit(enemyImg, (enemyX, enemyY))

# Game loop
running = True
while running:
    screen.fill((255, 51, 153))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    #if keystroke is pressed check whether its right or left   
        if  event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                rect1.x -= vel
                playerX_change -= 0.3
                gunX_change -= 0.3
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                rect1.x += vel
                playerX_change += 0.3
                gunX_change += 0.3
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                rect1.y -= vel
                playerY_change -= 0.3
                gunY_change -= 0.3
            if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                rect1.y += vel
                playerY_change += 0.3
                gunY_change += 0.3
            if event.key == pygame.K_SPACE:
                fire_bullet(gunX, gunY)
                mixer.music.load("sources/noises/laser.wav")
                mixer.music.play(1)
            
     #So lets start with colisio-
            vel = 10

        
        #checking whether key not press . death cannot come soon enough
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN or event.key == pygame.K_s or event.key == pygame.K_d or event.key == pygame.K_a or event.key  == pygame.K_w:
             playerX_change = 0
             playerY_change = 0    
                      
    #Player cant go out           
    playerX += playerX_change
    if playerX <= 0:
        playerX = 0
    elif playerX >= 736:
        playerX = 736
    elif playerY <= 0:
        playerY= 0
    elif playerY >= 536:
        playerY = 536
    

    #enemy dosent glitch out
    if enemyX <= 0:
        enemyX = 0
    elif enemyX >= 736:
        enemyX = 736
    elif enemyY <= 0:
        enemyY = 0
    elif enemyY >= 536:
        enemyY = 536

    #bullet destroy on fricking BOUNDARIE
    if bulletX >= 736:
        bulletState= "ready"

    #gun stay in place
    if gunY <= playerY:
        gunY = playerY
    elif gunY >= playerY:
        gunY = playerY

    # Bullet Movement
    if bulletX <= playerX:
        bulletX = gunX
        bulletState = "ready"

    if bulletState is "fire":
        fire_bullet(bulletX, playerY)
        bulletX += bulletY_change

    #spawnlocation
    pygame.draw.rect(screen, (255,255,255),spawnrect)

    playerY += playerY_change
    gunX += playerX_change
    gunY += playerY_change
    enemy()
    gun()
    player(playerX, playerY)
    pygame.display.update()