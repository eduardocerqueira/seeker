#date: 2022-11-29T17:07:53Z
#url: https://api.github.com/gists/fe1f24db2beb7edc83454058abc31c21
#owner: https://api.github.com/users/doke2

#!/usr/bin/python 
#
# $Id: xmas.py,v 1.1 2022/11/29 17:06:28 doke Exp $

import time
import board
import neopixel


# The order of the pixel colors - RGB or GRB. Some NeoPixels have red and green reversed!
# For RGBW NeoPixels, simply change the ORDER to RGBW or GRBW.
ORDER = neopixel.GRBW


color_seq = [ 
    ( 0x00, 0xff, 0x00, 0x00 ),  # red
    ( 0xff, 0x00, 0x00, 0x00 ),  # green
    ( 0x00, 0x00, 0x00, 0x00 ),  # black
    ]

pixels_per_color = 25 

delay = 0.1


pixel_pin = board.D18

# The number of NeoPixels
num_pixels = 248


pixels = neopixel.NeoPixel(
    pixel_pin, num_pixels, brightness=0.5, auto_write=False, pixel_order=ORDER
    )
pixelbuf = [ 0 ] * num_pixels

len_block = len( color_seq ) * pixels_per_color

def color_setup():
    for p in range( num_pixels ):
        block = p // len_block 
        color = ( p % len_block ) // pixels_per_color;
        pixelbuf[ p ] = color 

def color_cycle():
    for offset in range( len_block ):
        for p in range( num_pixels ):
            pixels[ ( offset + p ) % num_pixels ] = color_seq[ pixelbuf[ p % num_pixels ] ]
        pixels.show()
        time.sleep( delay )

starttime = time.time()
 
color_setup()
while True:
    color_cycle()  
    now = time.time()
    if now > starttime + 3600 * 3: 
        break

pixels.deinit()
