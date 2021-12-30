#date: 2021-12-30T16:55:29Z
#url: https://api.github.com/gists/3f7659cfd71244977f6cd61764da0fd7
#owner: https://api.github.com/users/linkfy

import pygame


if __name__ == "__main__":

    #Initialize
    
    pygame.init()
    screen = pygame.display.set_mode((200, 200))
    pygame.display.set_caption('Basic Pygame')
    
    #Fill back
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0,0,0))

    #Resources

    chat = pygame.image.load("img\\2.png")
    photo = pygame.image.load("img\\5.png")
    net = pygame.image.load("img\\28.png")
    off =  pygame.image.load("img\\77.png")
    on =  pygame.image.load("img\\78.png")
    circle = pygame.image.load("img\\107.png")
    msg = pygame.image.load("img\\71.png")
    msg = pygame.transform.scale(msg, (50,40))
    redDot = pygame.image.load("img\\111.png")
    blueDot = pygame.image.load("img\\110.png")
    block = pygame.image.load("img\\37.png")
    block = pygame.transform.scale(block, (30,40))
    

    #Switcher Parameters
    switcher = on
    switcher_pos = (20,80)
    switcher_status = True
    switcher_rect = switcher.get_rect()
    switcher_rect.x = switcher_pos[0]
    switcher_rect.y = switcher_pos[1]

    dot = redDot
    dot_status = True
    
    clock = pygame.time.Clock()

    start_ticks = pygame.time.get_ticks()
    
    while(True):
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x,y = event.pos
                if(switcher_rect.collidepoint(x,y)):
                    if(switcher_status):
                        switcher = off
                        switcher_status = False
                        print("OFF")
                    else:
                        switcher = on
                        switcher_status = True
                        print("ON")

        
        #Switch dot
        seconds = (pygame.time.get_ticks() - start_ticks)/10
        if(seconds % 2 == 0):
            dot_status = not dot_status
        if(dot_status):
            dot = blueDot
        else:
            dot = redDot
        
        
        #Blit & Flip
        screen.blit(background, (0,0))
        pygame.draw.rect(screen, (50, 234, 228), (0,0,200,200),5,50)
        
        
        
        screen.blit(chat, (20,20))
        screen.blit(photo, (75,20))
        screen.blit(net, (130,20))
        screen.blit(switcher, switcher_pos)
        screen.blit(circle, (80,80))
        screen.blit(msg, (135, 135))
        screen.blit(dot, (175, 140))
        screen.blit(block, (25, 130))
        pygame.display.flip()
