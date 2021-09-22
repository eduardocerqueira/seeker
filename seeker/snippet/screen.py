#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# screen.py

import kandinsky

import color

width, height = 320, 240 # screenWidth and screenHeight

def fill(color):
    kandinsky.fill_rect(0, 0, width, height, color)

def clear():
    fill(color.WHITE)
