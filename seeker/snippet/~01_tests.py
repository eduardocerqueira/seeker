#date: 2024-01-05T16:58:13Z
#url: https://api.github.com/gists/0c786468f6d0385d3f46d7c1a1d0aa99
#owner: https://api.github.com/users/florentbr

from micropython import const
from machine import Pin, mem32
from time import ticks_us, ticks_diff, sleep_ms
from esp32_pwm import PWM
import gc

GPIO_IN_REG = const(0x3FF4403C)

@micropython.viper
def wait_pin(gpio_mask: int, level: int) -> int:
    m = gpio_mask * level
    while (int(mem32[GPIO_IN_REG]) & gpio_mask) != m: pass
    return int(ticks_us())

def print_duties(pin_num, n):
    times = [ wait_pin(1 << pin_num, i & 1) for i in range(2 + n * 2) ]
    diffs = [ ticks_diff(b, a) for a, b in zip(times[1:-1], times[2:])]
    duties = [ '{:.1f}'.format(a * 100 / (a + b)) for a, b in zip(diffs[:-1:2], diffs[1::2])]
    print("duty%", duties)
    print("freq:", len(diffs) / sum(diffs) / 2 * 1e6)


PIN_16 = const(16)
PIN_17 = const(17)
PWM_MAX = const(0xffff)

pwm = PWM(Pin(PIN_16), freq = 50)

gc.collect()
pwm.duty_u16(PWM_MAX * 80//100, ramp = 50)
print_duties(PIN_16, 60)

gc.collect()
pwm.duty_u16(PWM_MAX * 20//100, ramp = 50)
print_duties(PIN_16, 60)