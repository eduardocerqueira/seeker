#date: 2023-04-28T17:04:31Z
#url: https://api.github.com/gists/662cbdf6694b2d3235d61d692828ad55
#owner: https://api.github.com/users/Goatghosts

import pygame
import sys
import pymunk
import pymunk.pygame_util

pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FRAMERATE = 60

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Physics with Pymunk")
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0, 1200)


def draw_background(screen):
    screen.fill((255, 255, 255))


class Character:
    def __init__(self, space, pos):
        self.space = space

        self.torso = pymunk.Body(10, pymunk.moment_for_box(10, (40, 80)))
        self.torso.position = pos
        torso_shape = pymunk.Poly.create_box(self.torso, (20, 60))
        space.add(self.torso, torso_shape)

        self.head = pymunk.Body(5, pymunk.moment_for_circle(5, 0, 20))
        self.head.position = (pos[0], pos[1] - 50)
        head_shape = pymunk.Circle(self.head, 20)
        space.add(self.head, head_shape)

        joint = pymunk.PivotJoint(self.torso, self.head, self.head.position)
        space.add(joint)


def draw_character(screen, character, options):
    for shape in character.torso.shapes.union(character.head.shapes):
        options.draw_shape(shape)


def main():
    character = Character(space, (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))

    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_shape = pymunk.Segment(ground_body, (0, WINDOW_HEIGHT - 20), (WINDOW_WIDTH, WINDOW_HEIGHT - 20), 10)
    space.add(ground_body, ground_shape)

    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = draw_options.flags ^ pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS
    draw_options.flags = draw_options.flags | pymunk.pygame_util.DrawOptions.DRAW_COLLISION_POINTS

    mouse_joint = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_position = pymunk.pygame_util.from_pygame(event.pos, screen)
                    shape = space.point_query_nearest(mouse_position, 0, pymunk.ShapeFilter())

                    if shape:
                        body = shape.shape.body
                        joint_point = pymunk.vec2d.Vec2d(mouse_position[0], mouse_position[1])
                        mouse_joint = pymunk.PivotJoint(body, space.static_body, joint_point)
                        mouse_joint.max_force = 500000
                        mouse_joint.collide_bodies = False
                        space.add(mouse_joint)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and mouse_joint:
                    space.remove(mouse_joint)
                    mouse_joint = None

            elif event.type == pygame.MOUSEMOTION and mouse_joint:
                mouse_position = pymunk.pygame_util.from_pygame(event.pos, screen)
                mouse_joint.anchor_b = mouse_position

        space.step(1 / 60)

        draw_background(screen)
        draw_character(screen, character, draw_options)
        space.debug_draw(draw_options)

        pygame.display.flip()
        clock.tick(FRAMERATE)


if __name__ == "__main__":
    main()
