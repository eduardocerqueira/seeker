#date: 2024-08-15T17:05:35Z
#url: https://api.github.com/gists/0db50f77309dab10ccb8f2d5260f59fe
#owner: https://api.github.com/users/dewmal

import pygame
import sys

# Constants
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 15
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
WHITE = (255, 255, 255)
FPS = 60

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")
clock = pygame.time.Clock()

# Game Variables
ball_pos = [WIDTH // 2, HEIGHT // 2]
ball_vel = [5, 5]
left_paddle_pos = [10, HEIGHT // 2 - PADDLE_HEIGHT // 2]
right_paddle_pos = [WIDTH - PADDLE_WIDTH - 10, HEIGHT // 2 - PADDLE_HEIGHT // 2]
left_score, right_score = 0, 0

def draw():
    screen.fill(WHITE)
    pygame.draw.rect(screen, (0, 0, 0), (*left_paddle_pos, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, (0, 0, 0), (*right_paddle_pos, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, (0, 0, 0), (ball_pos[0], ball_pos[1]), BALL_RADIUS)
    font = pygame.font.Font(None, 74)
    text = font.render(str(left_score), True, (0, 0, 0))
    screen.blit(text, (WIDTH // 4, 10))
    text = font.render(str(right_score), True, (0, 0, 0))
    screen.blit(text, (WIDTH * 3 // 4, 10))
    pygame.display.flip()

def handle_collision():
    global ball_vel, left_score, right_score
    if ball_pos[1] <= BALL_RADIUS or ball_pos[1] >= HEIGHT - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    if (left_paddle_pos[1] <= ball_pos[1] <= left_paddle_pos[1] + PADDLE_HEIGHT and
        ball_pos[0] - BALL_RADIUS <= left_paddle_pos[0] + PADDLE_WIDTH):
        ball_vel[0] = -ball_vel[0]
    if (right_paddle_pos[1] <= ball_pos[1] <= right_paddle_pos[1] + PADDLE_HEIGHT and
        ball_pos[0] + BALL_RADIUS >= right_paddle_pos[0]):
        ball_vel[0] = -ball_vel[0]
    if ball_pos[0] < 0:
        right_score += 1
        reset_ball()
    elif ball_pos[0] > WIDTH:
        left_score += 1
        reset_ball()

def reset_ball():
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    ball_vel = [5 * (-1 if left_score > right_score else 1), 5]

def move_ball():
    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]

# Main Game Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and left_paddle_pos[1] > 0:
        left_paddle_pos[1] -= 5
    if keys[pygame.K_s] and left_paddle_pos[1] < HEIGHT - PADDLE_HEIGHT:
        left_paddle_pos[1] += 5
    if keys[pygame.K_UP] and right_paddle_pos[1] > 0:
        right_paddle_pos[1] -= 5
    if keys[pygame.K_DOWN] and right_paddle_pos[1] < HEIGHT - PADDLE_HEIGHT:
        right_paddle_pos[1] += 5
    move_ball()
    handle_collision()
    draw()
    clock.tick(FPS)