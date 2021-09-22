#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# color.py

import kandinsky

BLACK    = kandinsky.color(0, 0, 0)
WHITE    = kandinsky.color(255, 255, 255)

BLUE     = kandinsky.color(0, 0, 255)
RED      = kandinsky.color(255, 0, 0)
GREEN    = kandinsky.color(0, 255, 0)

CYAN     = kandinsky.color(0, 255, 255)
MAGENTA  = kandinsky.color(255, 0, 255)
YELLOW   = kandinsky.color(255, 255, 0)

# color treatment

def grayscale(color):
    lightness = (0.21 * color[0]) + (0.72 * color[1]) + (0.07 * color[2])
    return kandinsky.color(lightness, lightness, lightness)       # ...

def invert(color):
    return kandinsky.color(255 - color[0], 255 - color[1], 255 - color[2])
