#date: 2022-03-10T17:04:53Z
#url: https://api.github.com/gists/29cc57a6cd6ff4c73f7fba116e1d5af8
#owner: https://api.github.com/users/rohit988


import pygame, sys, random
clock = pygame.time.Clock()
def size_setting():
    global ball
    pygame.draw.aaline(screen, color, (screen_width / 2, 0), (screen_width / 2, screen_height))
    pygame.draw.ellipse(screen, (255, 255, 0), ball)
    pygame.draw.rect(screen, color, player)
    pygame.draw.rect(screen, color, opponent)
def ball_movement():
    global ball, ball_y_speed, ball_x_speed

    if ball.right >= screen_width or ball.left <= 0:
        ball.x = screen_width/2
        ball.y = screen_height/2
    if ball.bottom >= screen_height or ball.top <= 0:
        ball_y_speed *=- 1
    ball.x += ball_x_speed
    ball.y += ball_y_speed
def player_movement():
    global player_speed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed += 10
            if event.key == pygame.K_UP:
                player_speed -= 10
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 10
            if event.key == pygame.K_UP:
                player_speed += 10
def wall_stop():
    global player,player_score,opponent_score
    if player.top <=0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height
    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        opponent.bottom = screen_height
    if ball.right >= screen_width:
        ball.x = screen_width/2
        ball.y = screen_height/2
        opponent_score +=1
    if ball.left <=0:
        ball.x = screen_width / 2
        ball.y = screen_height / 2
        player_score+=1
def collision():
    collision_value = 15
    if ball.colliderect(player):
        global ball_x_speed,ball_y_speed
        if abs(ball.right - player.left) <collision_value:
            ball_x_speed *= -1
        if abs(ball.top - player.bottom) < collision_value:
            ball_y_speed *= -1
            ball_x_speed *= -1
        if abs(ball.bottom - player.top) <collision_value:
            ball_y_speed *= -1
            ball_x_speed *= -1

    if ball.colliderect(opponent):
        if abs(ball.left - opponent.right)<=10:
            ball_x_speed*=-1
def opposite_player():
    if ball.top > opponent.top:
        opponent.top += opponent_speed
    if ball.top < opponent.top:
        opponent.top -= opponent_speed
ball_x_speed = 10 * random.choice((1, -1))
ball_y_speed = 10 * random.choice((1,-1))
player_speed = 0
screen_width = 1400
screen_height = 800
pygame.init()
ball = pygame.Rect(screen_width / 2 - 30, screen_height / 2 - 30, 60, 60)
player = pygame.Rect(screen_width - 40, screen_height / 2 - 70, 10, 140)
player.inflate(20,20)
opponent = pygame.Rect(30, screen_height / 2 - 70, 10, 140)
color = (250, 250, 250)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Ping Pong')
opponent_speed = 10
player_score = 0
opponent_score = 0
text_font = pygame.font.Font('freesansbold.ttf',32)


def start():
    while True:
        screen.fill((0, 0, 0))
        size_setting()
        ball_movement()
        opposite_player()
        player_movement()
        player.y += player_speed
        collision()
        wall_stop()
        player_text = text_font.render(f'{player_score}',True,'grey')
        screen.blit(player_text,(screen_width/2+10,screen_height/2))
        opponent_text = text_font.render(f'{opponent_score}', True, 'grey')
        screen.blit(opponent_text, (screen_width / 2 - 27, screen_height / 2))
        pygame.display.flip()
        clock.tick(60)
start()

