#date: 2023-05-02T16:47:42Z
#url: https://api.github.com/gists/c6a12bc3fa373f4a61715261326636de
#owner: https://api.github.com/users/NGC-1432

import board
import neopixel

pixel_pin = board.D2
num_pixels = 16

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

while True:
    for num in range(0, 254, 2):  # fade in loop no. 1
        COLOR = (int(num / 3), 0, int(num / 2))
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 1
        COLOR = (int(num / 3), 0, int(num / 2))
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 2
        COLOR = (0, 0, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 2
        COLOR = (0, 0, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 3
        COLOR = (0, num, 0)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 3
        COLOR = (0, num, 0)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 4
        COLOR = (num, 0, 0)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 4
        COLOR = (num, 0, 0)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 5
        COLOR = (num, num, 0)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 5
        COLOR = (num, num, 0)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 6
        COLOR = (num, 0, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 6
        COLOR = (num, 0, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 7
        COLOR = (num, num, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 7
        COLOR = (num, num, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 8
        COLOR = (0, num, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 8
        COLOR = (0, num, num)
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(0, 254, 2):  # fade in loop no. 9
        COLOR = (num, 0, int(num / 2))
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()
    for num in range(254, 0, -2):  # fade out loop no. 9
        COLOR = (num, 0, int(num / 2))
        for light in range(0, 16, 1):
            pixels[light] = COLOR
        pixels.show()