#date: 2021-12-30T16:59:00Z
#url: https://api.github.com/gists/d49a1a41f19fc7acc69a621ed9daa774
#owner: https://api.github.com/users/LionsPhil

import plasma
from plasma import plasma2040
from pimoroni import RGBLED, Button, Analog

import math
import random
import utime

sense = Analog(plasma2040.CURRENT_SENSE, plasma2040.ADC_GAIN, plasma2040.SHUNT_RESISTOR)

led = RGBLED(plasma2040.LED_R, plasma2040.LED_G, plasma2040.LED_B)
led.set_rgb(15, 0, 0)

button_a = Button(plasma2040.BUTTON_A)
button_b = Button(plasma2040.BUTTON_B)
button_boot = Button(plasma2040.USER_SW)

NUM_LEDS = 96 # Chunky diffuser
#NUM_LEDS = 332 # Ultra-dense
#NUM_LEDS = 96 + 332 # Both daisy chained

led_strip = plasma.WS2812(NUM_LEDS, 0, 0, plasma2040.DAT)
led_strip.start()

# "framebuffer" to composite effects onto
ledbuf = [(0, 0, 0)] * NUM_LEDS

def additive(led, r, g, b):
    if led < 0 or led >= NUM_LEDS:
        return
    ledbuf[led] = (
        int(min(ledbuf[led][0] + r, 255)),
        int(min(ledbuf[led][1] + g, 255)),
        int(min(ledbuf[led][2] + b, 255)))

class Sparkle():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.position = random.uniform(-1.0, NUM_LEDS+1.0)
        self.intensity = random.uniform(0.5, 1.0)
        self.momentum = random.uniform(-0.2, 0.2)
    
    def simulate(self):
        self.position += self.momentum
        self.intensity *= 0.9
        if self.position < -1.0 or self.position > NUM_LEDS + 1 or self.intensity < 1.0/255.0:
            self.reset()
    
    def render(self):
        leftpos = math.floor(s.position)
        rightfrac = s.position - leftpos
        leftintensity = s.intensity * 255 * (1.0 - rightfrac)
        rightintensity = s.intensity * 255 * rightfrac
        additive(leftpos, leftintensity * 2.0, leftintensity, 0)
        additive(leftpos + 1, rightintensity * 2.0, rightintensity, 0)

sparkles = []
for i in range(NUM_LEDS / 10):
    sparkles.append(Sparkle())

# https://lodev.org/cgtutor/plasma.html
# but it's just simple sine waves
class PlasmaWave():
    def __init__(self, frequency, r, g, b):
        self.frequency = frequency # random.uniform(math.pi * 4.0, math.pi * 8.0)
        self.phase = random.uniform(0.0, math.pi)
        self.momentum = random.uniform(0.1, 0.5)
        self.r = r
        self.g = g
        self.b = b
        if random.uniform(0.0, 1.0) > 0.5:
            self.momentum *= -1.0
        
    def simulate(self):
        self.phase += self.momentum
    
    def render(self):
        for i in range(NUM_LEDS):
            intensity = 0.5 * (1.0 + math.sin(((self.frequency * i) / NUM_LEDS) + self.phase))
            r = self.r * intensity
            g = self.g * intensity
            b = self.b * intensity
            additive(i, r, g, b)

plasma_waves = [
    PlasmaWave(math.pi * 1.0, 0, 16, 0),
    PlasmaWave(math.pi * 4.0, 0, 32, 0),
    PlasmaWave(math.pi * 8.0, 0, 32, 0)]
#for i in range(4):
#    plasma_waves.append(PlasmaWave())

SLEEP = 0.02

tick = 0
while True:
    # Blank buffer
    for i in range(NUM_LEDS):
        # 22 green is the threshold where the LED isn't quite off,
        # which looks better rather than having blank spans
        ledbuf[i] = (0, 22, 0)

    # Do some background greenery using plasma effects
    for w in plasma_waves:
        w.simulate()
        w.render()
        
    # Simulate and render sparkles
    for s in sparkles:
        s.simulate()
        s.render()
    
    # Blit, sort of
    for i in range(NUM_LEDS):
        led_strip.set_rgb(i, ledbuf[i][0], ledbuf[i][1], ledbuf[i][2])
    
    # Measure current and sleep
    tick += 1
    if tick > 10:
        print("Current =", sense.read_current(), "A")
        tick -= 10
    utime.sleep(SLEEP)
