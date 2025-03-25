#date: 2025-03-25T16:54:45Z
#url: https://api.github.com/gists/78cd7f639b1e723d0caf4ff697fde60c
#owner: https://api.github.com/users/12emmasullivan

import time
import analogio
import board
import neopixel
import math


dial_pin = board.A2 #pin 0 is Analog input 2 
dial = analogio.AnalogIn(dial_pin)

pixel_pin = board.D2
num_pixels = 16
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.5, auto_write=False)
 
while True:
    time.sleep(0.2)
    #print((math.sqrt(dial.value,)))
    val = int(math.sqrt(dial.value) - 20) #scale down potentiometer values to fit within color range
    print(val)
    if (val<20):
        val = 0