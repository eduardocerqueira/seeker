#date: 2021-10-18T17:17:32Z
#url: https://api.github.com/gists/ac0758ac77d6e20424253aea9deaeccc
#owner: https://api.github.com/users/voglster

from math import cos, radians, sin
from random import randint

import pygame
from easing_functions import QuadEaseInOut

# from maze_test.sprite_sheet import SpriteSheet

FPS = 30
gravity = (0, 0.3)


class Particle:
    def __init__(self, position, angle, speed, decay=5, color=None):
        self.position = position
        self.original_color = color or (255, 255, 255)
        self.age = 0
        angle = radians(angle)
        self.vector = (cos(angle) * speed, sin(angle) * speed)
        self.decay = decay
        self.size = randint(1, 7)

    @property
    def draw_position(self):
        return tuple(int(x) for x in self.position)

    @property
    def color(self):
        return tuple(max(0, self.original_color[i] - self.age) for i in range(3))

    def update(self):
        self.vector = tuple(sum(x) for x in zip(self.vector, gravity))
        self.position = (
            self.position[0] + self.vector[0],
            self.position[1] + self.vector[1],
        )
        self.age += self.decay

    @property
    def dead(self):
        return self.color == (0, 0, 0)


class World:
    def __init__(self, size):
        self.size = size
        self.points = []

    def spawn(self, position):
        color = tuple(randint(100, 255) for _ in range(3))
        count = randint(5, 50)
        rounds = randint(1, 5)
        initial = randint(0, 360)
        for i in range(count):
            angle = initial + ((360 / count * rounds) * i)
            self.points.append(
                Particle(position, angle, randint(2, 12), randint(2, 7), color=color)
            )

    def update(self):
        for point in tuple(self.points):
            point.update()
            if point.dead:
                self.points.remove(point)


class App:
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(200, 150)
        for j in [
            pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())
        ]:
            j.init()

        self._running = True
        self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Simulation")
        # ss = SpriteSheet()

        # self.block_surf = ss.load_image("block")
        # self.block_size = self.block_surf.get_size()
        # self.player_surf = ss.load_image("player")
        # self.goal_surf = ss.load_image("coin-01")
        # self.dot_surf = ss.load_image("dot")
        self.clock = pygame.time.Clock()
        self.world = World(pygame.display.get_surface())
        self.shooting = False

        self.ease = QuadEaseInOut(start=0, end=300, duration=2 * FPS)
        self.time = 0

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                self._running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.shooting = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.shooting = False

    def on_render(self):
        self.display.fill([1, 1, 1])
        self.time += 1
        for point in self.world.points:
            pygame.draw.circle(
                self.display, point.color, point.draw_position, point.size
            )
            # self.display.set_at(point.draw_position, point.color)

        if self.shooting:
            pos = pygame.mouse.get_pos()
            self.world.spawn(pos)

        pygame.display.flip()
        self.clock.tick(FPS)
        self.world.update()

    def on_execute(self):
        while self._running:
            pygame.event.pump()
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        pygame.quit()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
