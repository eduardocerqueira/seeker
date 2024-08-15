#date: 2024-08-15T17:07:47Z
#url: https://api.github.com/gists/bf291fbae85e31431a4eb91360aa092f
#owner: https://api.github.com/users/dewmal

import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
FPS = 60
BALL_SPEED_X, BALL_SPEED_Y = 5, 5
PADDLE_SPEED = 10

# Create the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong Game")

# Define paddle and ball
paddle_width, paddle_height = 10, 100
ball_size = 20

# Paddle positions
paddle_left = pygame.Rect(30, (HEIGHT - paddle_height) // 2, paddle_width, paddle_height)
paddle_right = pygame.Rect(WIDTH - 40, (HEIGHT - paddle_height) // 2, paddle_width, paddle_height)

# Ball position
ball = pygame.Rect((WIDTH // 2 - ball_size // 2), (HEIGHT // 2 - ball_size // 2), ball_size, ball_size)
ball_speed_x, ball_speed_y = BALL_SPEED_X, BALL_SPEED_Y

# Game loop
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get keys pressed
    keys = pygame.key.get_pressed()
    
    # Move paddles
    if keys[pygame.K_w] and paddle_left.top > 0:
        paddle_left.y -= PADDLE_SPEED
    if keys[pygame.K_s] and paddle_left.bottom < HEIGHT:
        paddle_left.y += PADDLE_SPEED
    if keys[pygame.K_UP] and paddle_right.top > 0:
        paddle_right.y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and paddle_right.bottom < HEIGHT:
        paddle_right.y += PADDLE_SPEED

    # Move ball
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with top and bottom
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y = -ball_speed_y

    # Ball collision with paddles
    if ball.colliderect(paddle_left) or ball.colliderect(paddle_right):
        ball_speed_x = -ball_speed_x

    # Ball out of bounds
    if ball.left <= 0 or ball.right >= WIDTH:
        ball.x = (WIDTH // 2 - ball_size // 2)
        ball.y = (HEIGHT // 2 - ball_size // 2)
        ball_speed_x = BALL_SPEED_X * (-1 if ball.left <= 0 else 1)
    
    # Clear the screen
    screen.fill(WHITE)
    
    # Draw paddles and ball
    pygame.draw.rect(screen, (0, 0, 0), paddle_left)
    pygame.draw.rect(screen, (0, 0, 0), paddle_right)
    pygame.draw.ellipse(screen, (0, 0, 0), ball)

    # Refresh display
    pygame.display.flip()
    clock.tick(FPS)