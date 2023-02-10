#date: 2023-02-10T16:46:20Z
#url: https://api.github.com/gists/72bc22ac4037e3b1f6278eccfb448781
#owner: https://api.github.com/users/robinkeum

from m5stack import *
from machine import Pin
from machine import PWM


led_pin = Pin(32,Pin.OUT)
#led_pwm = PWM((Pin)32)
button_pin = Pin(26, Pin.IN, Pin.PULL_UP)


# == compares the value of () to 0
while(True):
    if (button_pin.value() == 0):
        led_pin.on()

    else:
         led_pin.off()
    wait_ms(100)