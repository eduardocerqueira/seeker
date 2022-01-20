#date: 2022-01-20T16:57:03Z
#url: https://api.github.com/gists/61aee518226b885fb9adb65931bd1183
#owner: https://api.github.com/users/tobias-machts

import random
import math
import pygame
from pygame import mixer
pygame.init()

#game Window
screen = pygame.display.set_mode((800, 600))

pygame.display.set_caption("Space Fighter")
icon = pygame.image.load("resources/images/space-invaders.png")
pygame.display.set_icon(icon)






def Game():
    background = pygame.image.load("resources/images/background.jpg")

    #Sound
    backgroundSound = mixer.music.load("resources/music/background/backgroundMusic.wav")
    mixer.music.set_volume(1)
    mixer.music.play(-1)
    bulletSound = mixer.Sound("resources/music/GunShotSnglShotIn PE1097906.mp3")
    bulletSound.set_volume(0.25)
    collisionSound = mixer.Sound("resources/music/EXPLOSION.mp3")
    collisionSound.set_volume(0.33)


    #player
    class Player:
        def __init__(self):
            self.pImg = pygame.image.load("resources/images/space-invaders-player.png")
            self.pY = 480
            self.pX = 370
            self.pDX = 0
            self.movespeed = 0.5

        def SpawnPlayer(self):
            screen.blit(self.pImg, (self.pX, self.pY))

    player = Player()



    #enemy
    def GetEnemyDesign():
        designId = random.randint(0, 4)
        if designId == 0:
            return pygame.image.load("resources/images/enemy1.png")
        elif designId == 1:
            return pygame.image.load("resources/images/enemy2.png")
        elif designId == 2:
            return pygame.image.load("resources/images/enemy3.png")
        elif designId == 3:
            return pygame.image.load("resources/images/enemy4.png")
        else:
            return pygame.image.load("resources/images/enemy5.png")

    enemyMinspeed = -1
    enemyMaxspeed = 1

    class Enemy:
        def __init__(self):
            self.eImg = GetEnemyDesign()
            self.eY = random.randint(50, 150)
            self.eX = random.randint(0, (800-64))
            self.eDX = random.uniform(enemyMinspeed, enemyMaxspeed)
            self.eDY = 40
            self.freezed = 1
            self.freezeCounter = 0

        def SpawnEnemy(self):
            screen.blit(self.eImg, (self.eX, self.eY))

    Monster = []
    start = 6
    for i in range(start):
        Monster.append(Enemy())




    #Bullet
    class Bullet:
        def __init__(self):
            self.bImg = pygame.image.load("resources/images/bullet.png")
            self.bX = 0
            self.bY = 445
            self.bDY = -0.5



        def FireBullet(self):
            screen.blit(self.bImg, (self.bX+16, self.bY))


    geschoss = []


    maxcooldown = 250
    cooldown = 0
    cooldownBoost = 1
    cooldownBoostCounter = 0

    def ShowCooldown():
        cooldownFont = pygame.font.Font("resources/fonts/IHATCS__.TTF", 10)
        if cooldown > 0:
            showCoodlown = cooldownFont.render(str(cooldown), True, (255, 0, 0))
            screen.blit(showCoodlown, ((player.pX + 25), (player.pY + 75)))
        else:
            showCoodlown = cooldownFont.render(str(cooldown), True, (0, 255, 0))
            screen.blit(showCoodlown, ((player.pX + 30), (player.pY + 75)))




    def Collision (eX, eY, bX, bY):
        distance = math.sqrt(math.pow((eX-bX),2)+math.pow((eY-bY),2))
        if distance < 50:
            return True
        else:
            return False



    #score
    score = 0
    font = pygame.font.Font("resources/fonts/IHATCS__.TTF", 32)
    textX = 10
    textY = 10

    def Score():
        showscore = font.render("Score: "+str(score)+" Points", True, (255, 255, 255))
        screen.blit(showscore, (textX, textY))

    #gameOver
    gameOver = False
    wait = 1000
    def GameOver():
        font = pygame.font.Font("resources/fonts/IHATCS__.TTF", 50)
        showtext = font.render("Game Over! Score: " + str(score) + " Points", True, (255, 255, 255))
        screen.fill((0, 0, 0))
        screen.blit(showtext, (100, 250))
        mixer.music.stop()
        nonlocal cooldown
        cooldown = 1000
        nonlocal gameOver
        gameOver = True


    class PowerUp:
        def __init__(self, posX, posY, id, img):
            self.posX = posX
            self.posY = posY
            self.id = id
            self.img = img
            self.deltaY = 0.25

        def SpawnPowerUp(self):
            screen.blit(self.img, (self.posX, self.posY))

        def CollisionPowerUp(self, bX, bY):
            distance = math.sqrt(math.pow((bX - self.posX), 2) + math.pow((bY - self.posY), 2))
            if distance < 50:
                return True
            else:
                return False

    def TryPowerUp(eX, eY):
        number = random.randint(1, 100)
        if number < 11:
            #halbe nachladezeit
            powerUp.append(PowerUp(eX, eY, 1, pygame.image.load("resources/images/lightning_Freepik.png")))
        elif number < 16:
            powerUp.append(PowerUp(eX, eY, 2, pygame.image.load("resources/images/freezing_Freepik.png")))
            #freeze Monsters
        elif number < 18:
            powerUp.append(PowerUp(eX, eY, 3, pygame.image.load("resources/images/nuclear-bomb_Freepik.png")))
            #spawn megabomb
        elif number < 28:
            powerUp.append(PowerUp(eX, eY, 4, pygame.image.load("resources/images/poison_Freepik.png")))
            # spawn dead
        elif number < 29:
            powerUp.append(PowerUp(eX, eY, 5, pygame.image.load("resources/images/maschinengewehr_falticons.png")))
            #keine Nachladezeit
    powerUp = []

    #game
    running = True
    while running:
        screen.blit(background, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                mixer.music.load("resources/music/menu.mp3")
                mixer.music.set_volume(0.25)
                mixer.music.play(-1)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player.pDX -= player.movespeed
                if event.key == pygame.K_RIGHT:
                    player.pDX += player.movespeed
                if event.key == pygame.K_SPACE or event.key == pygame.K_s:
                    if cooldown == 0:
                        geschoss.append(Bullet())
                        geschoss[len(geschoss)-1].bX = player.pX
                        geschoss[len(geschoss)-1].FireBullet()
                        bulletSound.play()
                        cooldown = int(float(maxcooldown * cooldownBoost))

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    player.pDX += player.movespeed
                if event.key == pygame.K_RIGHT:
                    player.pDX -= player.movespeed

        player.pX += player.pDX

        if player.pX < 0:
            player.pX = 0
        elif player.pX > (800-64):
            #player ist 64 pixel breit --> breite-64
            player.pX = (800-64)

        #powerUp
        for i in range(len(powerUp)):
            #pass
            if powerUp[i].CollisionPowerUp(player.pX, player.pY):
                if powerUp[i].id == 1 and cooldownBoost > 0.5:
                    cooldownBoost = 0.5
                    cooldownBoostCounter = 1000
                if powerUp[i].id == 2:
                    for x in range(len(Monster)):
                        Monster[x].freezed = 0.25
                        Monster[x].freezeCounter = 1000
                if powerUp[i].id == 3:
                    for x in range(len(Monster)):
                        score += 1
                        Monster[x] = Enemy()
                        if score % 10 == 0:
                            Monster.append(Enemy())
                        if score % 50 == 0:
                            enemyMinspeed = enemyMinspeed * 2
                            enemyMaxspeed = enemyMaxspeed * 2
                            player.movespeed = player.movespeed * 2
                            if player.pDX < 0:
                                player.pDX = -player.movespeed
                            elif player.pDX > 0:
                                player.pDX = player.movespeed
                            maxcooldown = int(float(maxcooldown * 0.75))


                if powerUp[i].id == 4:
                    for x in range(len(Monster)):
                        Monster[x].eY = 500
                if powerUp[i].id == 5:
                    cooldownBoost = 0
                    cooldownBoostCounter = 1000


                powerUp[i].posY = 700
            else:
                powerUp[i].SpawnPowerUp()
                powerUp[i].posY = powerUp[i].posY + powerUp[i].deltaY

        #enemymovement
        for i in range(len(Monster)):

           #game over
            if Monster[i].eY >= 400:
                for a in range(len(Monster)):
                    Monster[a].eY = 2000
                GameOver()
                break

            #enemy Movement
            Monster[i].eX += Monster[i].eDX*Monster[i].freezed

            #freezedreset
            if Monster[i].freezeCounter > 0:
                Monster[i].freezeCounter -= 1
                Monster[i].freezed = 0.25
                screen.blit(pygame.image.load("resources/images/freezing_Freepik.png"), (760, 20))
            if Monster[i].freezeCounter == 0:
                Monster[i].freezed = 1

            if Monster[i].eX < 0 or Monster[i].eX > (800-64):
                Monster[i].eDX = Monster[i].eDX*(-1)
                Monster[i].eY += Monster[i].eDY
            # collision
            for bulletCounter in range(len(geschoss)):
                if Collision(Monster[i].eX, Monster[i].eY, geschoss[bulletCounter].bX, geschoss[bulletCounter].bY):
                    collisionSound.play()
                    geschoss[bulletCounter].bY = -10
                    score += 1
                    #power Up
                    TryPowerUp(Monster[i].eX, Monster[i].eY)
                    #neuen Gegner
                    Monster[i] = Enemy()
                    # neue Monster bei steigendem Score hinzufügen
                    if score % 10 == 0:
                        Monster.append(Enemy())
                    #neue monster bei hohem score verschnellern
                    if score % 50 == 0:
                        enemyMinspeed = enemyMinspeed*2
                        enemyMaxspeed = enemyMaxspeed*2
                        player.movespeed = player.movespeed*2
                        if player.pDX < 0:
                            player.pDX = -player.movespeed
                        elif player.pDX > 0:
                            player.pDX = player.movespeed
                        maxcooldown = int(float(maxcooldown*0.75))





                geschoss[bulletCounter].FireBullet()
                geschoss[bulletCounter].bY += geschoss[bulletCounter].bDY


        counter = 0
        while counter < len(geschoss):
            if geschoss[counter].bY <= 0:
                del geschoss[counter]
                counter = -1
            counter+= 1

        counter  = 0
        while counter < len(powerUp):
            if powerUp[counter].posY >= 650:
                del powerUp[counter]
                counter = -1
            counter+= 1

        for i in range(len(Monster)):
            Monster[i].SpawnEnemy()

        if not gameOver:
            player.SpawnPlayer()
            Score()
            ShowCooldown()
            if cooldown > 0:
                cooldown -=1
            if cooldownBoostCounter > 0:
                cooldownBoostCounter -= 1
                if cooldownBoost > 0:
                    screen.blit(pygame.image.load("resources/images/lightning_Freepik.png"), (730, 20))
                    screen.blit(pygame.image.load("resources/images/lightning-small.png"), ((player.pX+10), (player.pY + 70)))
                else:
                    screen.blit(pygame.image.load("resources/images/maschinengewehr_falticons.png"), (720, 20))
                    screen.blit(pygame.image.load("resources/images/maschinengewehr_small.png"), ((player.pX+10), (player.pY + 70)))
            if cooldownBoostCounter == 0:
                cooldownBoost = 1


        else:
            if wait > 0:
                wait -= 1
                waitText = pygame.font.Font("resources/fonts/IHATCS__.TTF", 10).render(str(wait), True, (255, 255, 255))
                screen.blit(waitText, (750, 550))
            else:
                running = False
                mixer.music.load("resources/music/menu.mp3")
                mixer.music.set_volume(0.25)
                mixer.music.play(-1)




        pygame.display.update()


















def Menu():
    background = pygame.image.load("resources/images/background.jpg")
    # mixer.music.set_volume(mixer.music.get_volume()*0.25)
    mixer.music.load("resources/music/menu.mp3")
    mixer.music.set_volume(0.25)
    mixer.music.play(-1)

    class Button:
        def __init__(self):
            self.img = pygame.image.load("resources/images/start.png")
            self.imgHover = pygame.image.load("resources/images/start-hover.png")
            self.posX = 336
            self.posY = 250
            self.posRealY = self.posY+35
            self.posEndeY = self.posRealY+58

        def DrawButton(self):
            screen.blit(self.img, (self.posX, self.posY))
        def DrawHoverButton(self):
            screen.blit(self.imgHover, (self.posX, self.posY))
    start = Button()

    def Text():
        font = pygame.font.Font("resources/fonts/IHATCS__.TTF", 50)
        showtext = font.render("Space Fighter", True, (255, 255, 255))
        screen.blit(showtext, (275, 175))
    def Anleitung():
        font = pygame.font.Font("resources/fonts/IHATCS__.TTF", 20)
        showtext = font.render("Press left/right arrow key to move; Press space (or 's') to shoot", True, (255, 255, 255))
        #screen.blit(showtext, (180, 475))
        screen.blit(showtext, (165, 475))
    def CreditButton(hover):
        font = pygame.font.Font("resources/fonts/IHATCS__.TTF", 20)
        if hover:
            showtext = font.render("Credits", True, (0, 255, 0))
        else:
            showtext = font.render("Credits", True, (255, 255, 255))
        screen.blit(showtext, (740, 575))
    running = True
    while running:
        screen.blit(background, (0, 0))
        Text()
        Anleitung()
        CreditButton(False)
        #screen.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                #mixer.music.stop()


        maus = pygame.mouse.get_pos()

        if maus[0]>start.posX and maus[0]<(start.posX+128) and maus[1]>start.posRealY and maus[1]<start.posEndeY:
            start.DrawHoverButton()
            if pygame.mouse.get_pressed()[0] == 1:
                #mixer.music.stop()
                Game()
        else:
            start.DrawButton()

        if maus[0]>740 and maus[0]<799 and maus[1]>575 and maus[1]<599:
            CreditButton(True)
            if pygame.mouse.get_pressed()[0] == 1:
                Credits()

        pygame.display.update()




















def Credits():
    background = pygame.image.load("resources/images/background.jpg")
    mixer.music.load("resources/music/Victory.mp3")
    mixer.music.set_volume(0.25)
    mixer.music.play(0)

    class Part:
        def __init__(self, text, img, eingerueckt):
            self.pos = 600
            self.text = text
            self.img = pygame.image.load(img)
            self.size = 40
            self.eingerueckt = eingerueckt

        def IconPart(self):
            if self.eingerueckt:
                screen.blit(self.img, (130, self.pos))
            else:
                screen.blit(self.img, (100, self.pos))
        def TextPart(self):
            font = pygame.font.Font("resources/fonts/IHATCS__.TTF", 20)
            showtext = font.render(self.text, True, (255,255,255))
            if self.eingerueckt:
                screen.blit(showtext, (330, self.pos+10))
            else:
                screen.blit(showtext, (300, self.pos+10))


    credits = []
    credits.append(Part("Developed by Tobias W. (Github: @tobias_machts)", "resources/images/credit/Eucalyp-developer.png", False))
    credits[0].size = credits[0].size * 1.5
    credits.append(Part("Font made by 'Utopia - Dale Harris' ...", "resources/images/credit/Roman_Káčerek-text.png", False))
    credits.append(Part("from 'www.dafont.com'", "resources/images/credit/nothing.png", False))
    credits.append(Part("Game music - Tracks made by 'Makai Symphony' ...","resources/images/credit/Freepik-musik.png", False))
    credits.append(Part("from 'www.chosic.com'", "resources/images/credit/nothing.png", False))
    credits.append(Part("Credit music made by 'Velimir Andreev' ...", "resources/images/credit/Freepik-musik.png", False))
    credits.append(Part("from 'www.storyblocks.com'", "resources/images/credit/nothing.png", False))
    credits.append(Part("Lobby music made by 'AleX Zavesa' from 'www.pixabay.com'","resources/images/credit/Freepik-musik.png", False))
    credits.append(Part("Shooting sound made by 'Videvo' from 'www.videvo.net'","resources/images/credit/Freepik-sound-effect.png", False))
    credits.append(Part("Explosion sound made by 'Administrator' ...","resources/images/credit/Freepik-sound-effect.png", False))
    credits.append(Part("from 'www.sfxbuzz.com'", "resources/images/credit/nothing.png", False))
    credits[10].size = credits[10].size*2
    credits.append(Part("All Icons are from 'www.flaticon.com'", "resources/images/credit/Freepik-icons.png", False))
    #credits[11].size = credits[11].size * 1.5
    credits.append(Part("Icon made by 'Eucalyp'", "resources/images/credit/Eucalyp-developer.png", True))
    credits.append(Part("Icon made by 'Roman Káčerek'", "resources/images/credit/Roman_Káčerek-text.png", True))
    credits.append(Part("Icon made by 'Smalllikeart'", "resources/images/credit/smalllikeart-enemy4.png", True))
    credits.append(Part("Icon made by 'Flat Icons'", "resources/images/maschinengewehr_falticons.png", True))
    credits.append(Part("Icon made by 'Smashicons'", "resources/images/credit/smashicons-player.png", True))
    credits.append(Part("Icon made by 'Smashicons'", "resources/images/credit/smashicons-raumschiff.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepik-start-hover.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepik-musik.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepik-sound-effect.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepix-bullet.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepix-enemy1.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepix-enemy2.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepix-enemy5.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepix-enemy3.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/lightning_Freepik.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/freezing_Freepik.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/nuclear-bomb_Freepik.png", True))
    credits.append(Part("Icon made by 'Freepik'", "resources/images/poison_Freepik.png", True))

    credits.append(Part("Icon made by 'Freepik'", "resources/images/credit/Freepik-icons.png", True))


    for i in range(1, len(credits)):
        credits[i].pos = credits[i-1].pos + credits[i-1].size


    running = True
    while running:
        screen.blit(background, (0, 0))
        # screen.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                mixer.music.load("resources/music/menu.mp3")
                mixer.music.set_volume(0.25)
                mixer.music.play(-1)

        for i in range(len(credits)):
            credits[i].TextPart()
            credits[i].IconPart()
            credits[i].pos -= 0.75

        if credits[len(credits)-1].pos < -50:
            running = False
            mixer.music.load("resources/music/menu.mp3")
            mixer.music.set_volume(0.25)
            mixer.music.play(-1)

        pygame.display.update()












Menu()
#pip install pyinstaller
#pip install auto-py-to-exe