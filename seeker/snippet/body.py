#date: 2022-10-03T17:26:48Z
#url: https://api.github.com/gists/2cbfaeaf1ab2b035be7f7809213e6651
#owner: https://api.github.com/users/xiaoxuan-fan

# Author: Xiaoxuan
# Date: 5/11/2021
# Purpose: create a Body class to simulate celestial bodies
from cs1lib import *


class Body:
    def __init__(self, mass, x, y, vx, vy, pixel_radius, r, g, b):
        self.mass = mass                  # kilograms
        self.x = x                        # meters
        self.y = y                        # meters
        self.vx = vx                      # meters per second
        self.vy = vy                      # meters per second
        self.pixel_radius = pixel_radius  # pixels
        self.r = r
        self.g = g
        self.b = b

    def update_position(self, timestep):
        self.x += self.vx * timestep
        self.y += self.vy * timestep

    def update_velocity(self, ax, ay, timestep):
        self.vx += ax * timestep
        self.vy += ay * timestep

    def draw(self, cx, cy, pixels_per_meter):
        coordinate_x = self.x * pixels_per_meter + cx
        coordinate_y = cy - self.y * pixels_per_meter
        set_fill_color(self.r, self.g, self.b)
        draw_circle(coordinate_x, coordinate_y, self.pixel_radius)
