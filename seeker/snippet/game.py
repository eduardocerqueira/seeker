#date: 2024-11-13T16:48:27Z
#url: https://api.github.com/gists/200b0f0e0977b1dcf2664703ad62f03b
#owner: https://api.github.com/users/ender18g

# Example file showing a basic pygame "game loop"
import pygame
from extra_stuff import build_background
from random import randint

# pygame setup
pygame.init()
WIDTH = 1280
HEIGHT = 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

class Ship(pygame.sprite.Sprite):
    def __init__(self, x,y, screen):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.velocity = 1
        self.image = pygame.image.load('assets/Ships/ship (14).png')
        self.rect = self.image.get_rect() 
        self.screen = screen
        self.screen_rect = self.screen.get_rect()
    
    def update(self):
        if not self.screen_rect.contains(self.rect):
            # if the ship is NOT in my screen
            self.velocity = -self.velocity
        
        if self.rect.x == 100:
            self.kill()

        # move the ship    
        self.x += self.velocity

        # update my rectangle
        self.rect.center = (self.x, self.y)


    
    def draw(self, screen):
        # blit the image on screen
        screen.blit(self.image, self.rect )


ship_group = pygame.sprite.Group()

# make lots of ships
for i in range(5):
    s =  Ship(randint(0,WIDTH), randint(0,HEIGHT), screen)
    # add the ship to a sprite group
    ship_group.add(s)






background = build_background(WIDTH, HEIGHT)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")

    # blit my background on the screen
    screen.blit(background, (0,0))

    # update the ship position
    ship_group.update()
    # draw my ship
    ship_group.draw(screen)

    # RENDER YOUR GAME HERE

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()