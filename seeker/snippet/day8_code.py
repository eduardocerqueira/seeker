#date: 2024-03-13T16:59:16Z
#url: https://api.github.com/gists/816f2098f8f29b22b270c125246b978c
#owner: https://api.github.com/users/ihlan48


import board
import neopixel
 
pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

while True:
    for num in range(0,254,2):   #fade in loop
        COLOR=(num,138,0)
        pixels[1]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,0,183)
        pixels[3]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,0,203)
        pixels[5]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,39,57)
        pixels[7]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,207,124)
        pixels[9]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,225,0)
        pixels[11]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,5,110)
        pixels[13]=COLOR
        pixels.show()
    for num in range(255,25,153):   #fade out loop
        COLOR=(num,0,0)
        pixels[15]=COLOR
        pixels.show()
        for num in range(278,178,209):   #fade out loop
        COLOR=(num,0,0)
        pixels[8]=COLOR
        pixels.show()