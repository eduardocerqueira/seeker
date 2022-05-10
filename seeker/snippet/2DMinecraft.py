#date: 2022-05-10T16:58:52Z
#url: https://api.github.com/gists/cc3ec1e62ca70bd5be3683202ad24dc9
#owner: https://api.github.com/users/Editor0ne

import random
import sys
import pygame
from pygame.display import set_caption
from pygame.locals import *

set_caption("2D Minecraft")
a = pygame.image.load('controller.png')
pygame.display.set_icon(a)
instructions = """
    Move with Left, Right, Up, and Down arrows
    Mine with the spacebar
    Place Grass with 1
    Place Stone with 2
    Place Water with 3
    
    Enjoy!
"""
print(instructions)
cloudx, cloudy = -200, 0
# declare resources
DIRT, GRASS, WATER, COAL, CLOUD, WOOD = 0, 1, 2, 3, 4, 5
# declare valuable resources
FIRE, SAND, GLASS, ROCK, STONE, BRICK, DIAMOND = 6, 7, 8, 9, 10, 11, 12

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PB_DISP_AREA = 50
TILESIZE = 40
MAPWIDTH = 30
MAPHEIGHT = 20
PickedBlocks = {DIRT: 0,
                FIRE: 0,
                SAND: 0,
                GLASS: 0,
                ROCK: 0,
                STONE: 0,
                BRICK: 0,
                DIAMOND: 0,
                WOOD: 0,
                CLOUD: 0,
                GRASS: 0,
                COAL: 0,
                WATER: 0
                }

# import an image for each of the resources
textures = {
    DIRT: pygame.image.load('dirt.png'),
    GRASS: pygame.image.load('grass.png'),
    WATER: pygame.image.load('water.png'),
    COAL: pygame.image.load('coal.png'),
    CLOUD: pygame.image.load('cloud.png'),
    BRICK: pygame.image.load('brick.png'),
    DIAMOND: pygame.image.load('diamond.png'),
    FIRE: pygame.image.load('fire.png'),
    GLASS: pygame.image.load('glass.png'),
    ROCK: pygame.image.load('rock.png'),
    SAND: pygame.image.load('sand.png'),
    STONE: pygame.image.load('stone.png'),
    WOOD: pygame.image.load('wood.png')
}

controls = {
    DIRT: K_1,
    GRASS: K_2,
    WATER: K_3,
    COAL: K_4,
    WOOD: K_5,
    FIRE: K_6,
    SAND: K_7,
    GLASS: K_8,
    ROCK: K_9,
    STONE: K_0,
    BRICK: K_b,
    DIAMOND: K_d
}

craft = {
    FIRE: {WOOD: 2, ROCK: 2},
    STONE: {ROCK: 2},
    GLASS: {FIRE: 1, SAND: 2},
    DIAMOND: {WOOD: 2, COAL: 3},
    BRICK: {ROCK: 2, FIRE: 1},
    SAND: {ROCK: 2}
}

player = pygame.image.load('minepickaxe3.png')
playerPos = [0, 0]
cross_pos = [0, 0]
crossImg = pygame.image.load("crosshair1.png")
resources = [DIRT, GRASS, WATER, COAL, WOOD, FIRE, SAND, GLASS, ROCK, STONE, BRICK, DIAMOND]
fpsClock = pygame.time.Clock()
# initialize the map with all dirt
tilemap = [[DIRT for w in range(MAPWIDTH)] for h in range(MAPHEIGHT)]

# for each row
for row in range(MAPHEIGHT):
    # for each column in that row
    for col in range(MAPWIDTH):
        rn = random.randint(0, 15)
        if rn == 0:
            tile = COAL
        elif rn in [1, 2]:
            tile = WATER
        elif rn in [3, 4, 5, 6, 7]:
            tile = GRASS
        elif rn in [7, 8, 9]:
            tile = WOOD
        elif rn in [9, 10, 11]:
            tile = ROCK
        else:
            tile = DIRT
        tilemap[row][col] = tile

pygame.init()
DISPLAYSURF = pygame.display.set_mode((MAPWIDTH * TILESIZE, MAPHEIGHT * TILESIZE + PB_DISP_AREA))
PBFONT = pygame.font.Font('freesansbold.ttf', 18)
while True:
    DISPLAYSURF.fill(BLACK)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_RIGHT and playerPos[0] < MAPWIDTH - 1:
                playerPos[0] += 1
            if event.key == K_LEFT and playerPos[0] > 0:
                playerPos[0] -= 1
            if event.key == K_DOWN and playerPos[1] < MAPHEIGHT - 1:
                playerPos[1] += 1
            if event.key == K_UP and playerPos[1] > 0:
                playerPos[1] -= 1
            if event.key == K_SPACE:
                currentTile = tilemap[playerPos[1]][playerPos[0]]
                PickedBlocks[currentTile] += 1
                tilemap[playerPos[1]][playerPos[0]] = DIRT
            for key in controls:
                if event.key == controls[key]:
                    if pygame.mouse.get_pressed()[0]:
                        if key in craft:
                            canBeMade = True
                            for i in craft[key]:
                                if craft[key][i] > PickedBlocks[i]:
                                    canBeMade = False
                                    break
                            if canBeMade:
                                for i in craft[key]:
                                    PickedBlocks[i] -= craft[key][i]
                                PickedBlocks[key] += 1
                    else:
                        currentTile = tilemap[playerPos[1]][playerPos[0]]
                        if PickedBlocks[key] > 0:
                            PickedBlocks[key] -= 1
                            PickedBlocks[currentTile] += 1
                            tilemap[playerPos[1]][playerPos[0]] = key

    for row in range(MAPHEIGHT):
        for column in range(MAPWIDTH):
            DISPLAYSURF.blit(textures[tilemap[row][column]], (column * TILESIZE, row * TILESIZE))

    DISPLAYSURF.blit(player, (playerPos[0] * TILESIZE, playerPos[1] * TILESIZE))

    DISPLAYSURF.blit(textures[CLOUD].convert_alpha(), (cloudx, cloudy))
    cloudx += 1
    if cloudx > MAPWIDTH * TILESIZE:
        cloudy = random.randint(0, MAPHEIGHT * TILESIZE)
        cloudx = -200

    placePosition = 10
    for item in resources:
        DISPLAYSURF.blit(textures[item], (placePosition, MAPHEIGHT * TILESIZE + 20))
        placePosition += 30
        textObj = PBFONT.render(str(PickedBlocks[item]), True, WHITE, BLACK)
        DISPLAYSURF.blit(textObj, (placePosition, MAPHEIGHT * TILESIZE + 20))
        placePosition += 50
    pygame.display.update()
    fpsClock.tick()
