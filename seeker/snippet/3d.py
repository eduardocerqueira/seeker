#date: 2025-05-22T16:49:29Z
#url: https://api.github.com/gists/9ece46e1f51e798dabb6fcd0a8aa06d5
#owner: https://api.github.com/users/scratcheurs25

from math import *
from kandinsky import *

camX = 0
camY = 0
camZ = 0
distanceToScreen = 200

def draw_pixel_3d(x, y, z, color):
    global distanceToScreen
    if z == 0:  # Évite la division par zéro
        raise ValueError("La profondeur z ne peut pas être zéro.")
    
    screenX = (x * distanceToScreen) / z
    screenY = (y * distanceToScreen) / z

    set_pixel(int(screenX), int(screenY), color)

# Exemple d'utilisation
draw_pixel_3d(0, 0, 400, (0, 0, 0))# This comment was added automatically to allow this file to save.
# You'll be able to remove it after adding text to the file.
