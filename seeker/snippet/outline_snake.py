#date: 2022-09-01T17:14:28Z
#url: https://api.github.com/gists/193ec6407393dc2035199fb47717e1fd
#owner: https://api.github.com/users/hitbox

import argparse

import random

from operator import attrgetter
from collections import Counter

import pygame

SIDES = ('top', 'right', 'bottom', 'left')
getsides = attrgetter(*SIDES)

SIDELINE = {
    'top': ('topleft', 'topright'),
    'right': ('topright', 'bottomright'),
    'bottom': ('bottomleft', 'bottomright'),
    'left': ('topleft', 'bottomleft'),
}

def sideline(rect, sidename):
    attrs = SIDELINE[sidename]
    return (getattr(rect, attrs[0]), getattr(rect, attrs[1]))

def allsidelines(rect):
    return tuple(sideline(rect, sidename) for sidename in SIDELINE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Format string for frame output.')
    args = parser.parse_args()

    clock = pygame.time.Clock()
    window = pygame.display.set_mode((20*40,)*2)
    frame = window.get_rect()
    snake = [pygame.Rect((0,20*10), (20,)*2) for _ in range(10)]

    for r1, r2 in zip(snake[:-1], snake[1:]):
        r2.left = r1.right

    head_velocity = (1, 0)
    velocities = [head_velocity for _ in snake]
    move = move_delay = 250
    frame = 0
    running = True
    while running:
        elapsed = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            head_velocity = (0, -1)
        elif pressed[pygame.K_RIGHT]:
            head_velocity = (1, 0)
        elif pressed[pygame.K_DOWN]:
            head_velocity = (0, 1)
        elif pressed[pygame.K_LEFT]:
            head_velocity= (-1, 0)
        move -= elapsed
        if move <= 0:
            move = move_delay
            for vel, sec in zip(velocities, snake):
                sec.x += vel[0] * sec.width
                sec.y += vel[1] * sec.height
            velocities = velocities[1:] + [head_velocity]
        window.fill('black')
        for rect in snake:
            pygame.draw.rect(window, 'ghostwhite', rect)
        # keep only unique sides
        lines = Counter(line for rect in snake for line in allsidelines(rect))
        lines = [line for line, count in lines.items() if count == 1]
        for line in lines:
            pygame.draw.line(window, 'red', line[0], line[1], 6)
        pygame.display.flip()
        if args.output:
            path = args.output.format(frame)
            pygame.image.save(window, path)
            frame += 1

if __name__ == '__main__':
    main()