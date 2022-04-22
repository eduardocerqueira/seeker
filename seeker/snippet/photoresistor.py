#date: 2022-04-22T17:11:32Z
#url: https://api.github.com/gists/37b3bb8a307cb5259b1fde38ad00b973
#owner: https://api.github.com/users/mpschneider2

import board
import digitalio
import time
import neopixel
from analogio import AnalogIn

pixels = neopixel.NeoPixel(board.D1, 30)
photoin = AnalogIn(board.A1)

def get_voltage(pin):
    return (pin.value * 3.3) / 65536

colors = [(255, 0, 0), #red
(255, 150, 0), #yellow
(0, 255, 0), #green
(0, 255, 255), #cyan
(0, 0, 255), #blue
(180, 0, 255), #purple,
(255,255,255) #white
]

while True:
    print(get_voltage(photoin))
    if get_voltage(photoin) < 3.1:
        pixels.fill(0)
        pixels.show()
    else:
        pixels.fill((255, 255, 255))
        pixels.show()