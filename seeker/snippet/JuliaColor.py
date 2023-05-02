#date: 2023-05-02T16:48:56Z
#url: https://api.github.com/gists/6e32cd27a00c03a23eaf7c1a87a8494b
#owner: https://api.github.com/users/justjuliafied

import board
import neopixel 
pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.9, auto_write=False)



while True:
    for num in range(0,254,2):   #fade in loop
        COLOR=(num,num,254)
        pixels.fill(COLOR)
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,num,254)
        pixels.fill(COLOR)
        pixels.show()