#date: 2022-07-01T17:12:36Z
#url: https://api.github.com/gists/658b3e57dc97398102a57ada76080dda
#owner: https://api.github.com/users/matheusls1

import pygame
import random

#iniciador do pygame
pygame.init()

#tamanho de tela
screen = pygame.display.set_mode((800, 600))

#plano de fundo
background = pygame.image.load('background.png')

#titulo e icone
pygame.display.set_caption("Space Invaders")
icon = pygame.image.load('ufo.png')
pygame.display.set_icon(icon)

#Player
playerImg = pygame.image.load('player.png')
playerX = 370
playerY = 480
playerX_change = 0


#enemy
enemyImg = pygame.image.load('enemy.png')
enemyX = random.randint(0, 800)
enemyY = random.randint(50, 150)
enemyX_change = 4
enemyY_change = 40

#bala
bulletImg = pygame.image.load('bullet.png ')
bulletX = 0
bulletY = 480
bulletX_change = 0
bulletY_change = 10
bullet_state = "ready"

def player(x, y):
    screen.blit(playerImg, (x, y))

def enemy(x, y):
    screen.blit(enemyImg, (x, y))

def fire_bullet(x, y):
    global  bullet_state
    bullet_state = "fire"
    screen.blit(bulletImg, (x + 16, y + 10))

#GAME LOOP
running = True
while running:

# RGB -Red - Green - Blue
   screen.fill((0, 0, 0))

#imagem do plano de fundo
screen.blit(background, (0, 0))

for event in pygame.event.get():
       if event.type == pygame.QUIT:
           running = False
       #o pressionamento de tecla for pressionado, verifique se está à direita ou à esquerda
       if event.type == pygame.KEYDOWN:

           if event.key == pygame.K_LEFT:
               playerX_change = - 5
           if event.key == pygame.K_RIGHT:
               playerX_change = 5
           if event.key == pygame.K_SPACE:
               fire_bullet(playerX,bulletY)


       if event.type == pygame.KEYUP:
           if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
               playerX_change = 5


#limite de tela onde player e enemy podem andar
playerX += playerX_change
if playerX <= 0:
    playerX = 0
elif playerX >= 736:
    playerX = 736

enemyX += enemyX_change
if enemyX <= 0:
    enemyX_change = 4
    enemyY += enemyX_change
elif enemyX >= 736:
    enemyX_change = -4
    enemyY += enemyX_change


#movimentaçao da bala
if bullet_state is  "fire":
    fire_bullet(playerX,bulletY)
    bulletY -= bulletY_change#####PAREI AQ

player(playerX, playerY)
enemy(enemyX, enemyY)
pygame.display.update()


