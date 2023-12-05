#date: 2023-12-05T17:08:31Z
#url: https://api.github.com/gists/8703b1a351108984a224e6db7880c80f
#owner: https://api.github.com/users/AspirantDrago

import pygame

SIZE = WIDTH, HEIGHT = 600, 400
BACKGROUND = pygame.Color('blue')
COLOR = pygame.Color('yellow')
FPS = 60

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(BACKGROUND)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
