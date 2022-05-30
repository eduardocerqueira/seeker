#date: 2022-05-30T16:52:25Z
#url: https://api.github.com/gists/75299ba898828ed89e204e9cbf94ef33
#owner: https://api.github.com/users/saewoonam

# Example using PIO to blink an LED and raise an IRQ at 1Hz.

import time
from machine import Pin
import rp2


@rp2.asm_pio(set_init=rp2.PIO.OUT_LOW)
def blink_1hz():
    pull(block)
    mov(x, osr)
    set(y,24)
    #irq(rel(0))
    label("loop")
    #irq(rel(0))
    nop()
    # Cycles: 1 + 1 + 6 + 32 * (30 + 1) = 1000
    set(pins, 1)
    mov(x, osr)                  
    label("delay_high")                       
    jmp(x_dec, "delay_high")
    nop()
    nop()
    # Cycles: 1 + 7 + 32 * (30 + 1) = 1000
    set(pins, 0)
    mov(x, osr)                
    label("delay_low")                       
    jmp(x_dec, "delay_low")
    jmp(y_dec, "loop")
    #irq(rel(0))

# Create the StateMachine with the blink_1hz program, outputting on Pin(25).
sm = rp2.StateMachine(0, blink_1hz, freq=2_000_000, set_base=Pin(0))
start = 0
# Set the IRQ handler to print the millisecond timestamp.
def handler(p):
    global start
    new = time.ticks_us()
    print(new, new-start)
    start = new
#sm.irq(lambda p: print(time.ticks_ms()))
sm.irq(handler)
# Start the StateMachine.
sm.put(35)
sm.active(1)
sm.put(34)
