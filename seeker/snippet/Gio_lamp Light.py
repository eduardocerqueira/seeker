#date: 2022-03-23T17:04:13Z
#url: https://api.github.com/gists/fbd1c93ab3b9cbcd91627839965b6a5f
#owner: https://api.github.com/users/GioLazo

import time
import analogio
import board
import neopixel
import math
import adafruit_dotstar

builtInLed = adafruit_dotstar.DotStar(board.APA102_SCK, board.APA102_MOSI, 1)
builtInLed.brightness = 0

dial_pin = board.A2 #pin 0 is Analog input 2 
dial = analogio.AnalogIn(dial_pin)

pixel_pin = board.D2
num_pixels = 12
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.5, auto_write=False)
 
while True:
    time.sleep(0.2)
    #print((math.sqrt(dial.value,)))
    val = int(math.sqrt(dial.value)-20) #scale down potentiometer values to fit within color range
    print(val)
    if(val>-5 and val<5):
        pixels.fill(
    pixels.fill((val,val,val))
    pixels.show()