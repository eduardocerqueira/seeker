#date: 2024-01-05T16:58:13Z
#url: https://api.github.com/gists/0c786468f6d0385d3f46d7c1a1d0aa99
#owner: https://api.github.com/users/florentbr

from micropython import const
from machine import Pin
from time import sleep
from esp32_pwm import PWM

PWM_MAX = const(0xffff)

pwm = PWM(Pin(16), freq = 50)

# duty to 80% with ramping over 50 cycles (1sec)
pwm.duty_u16(PWM_MAX * 80//100, ramp = 50)

sleep(2)

# duty to 20% with ramping over 50 cycles (1sec)
pwm.duty_u16(PWM_MAX * 20//100, ramp = 50)