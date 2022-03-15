#date: 2022-03-15T17:02:01Z
#url: https://api.github.com/gists/fe940587ac9a15ff7cab4c85ddee8676
#owner: https://api.github.com/users/benevpi


#Note this is in Micropython not circuit python as circuit python doesn't give enough control over the state machines.

from machine import Pin, ADC
import array
from rp2 import PIO, StateMachine, asm_pio
import uctypes
from uctypes import BF_POS, BF_LEN, UINT32, BFUINT32, struct
from math import floor, ceil
from time import sleep
import _thread

min_sound_freq = 110 #A2
max_sound_freq = 7040 #A8


@asm_pio(out_init=(PIO.OUT_HIGH, PIO.OUT_LOW,PIO.OUT_LOW,PIO.OUT_LOW,PIO.OUT_LOW,PIO.OUT_LOW,PIO.OUT_LOW,), out_shiftdir=PIO.SHIFT_RIGHT)
def seven_bit_dac(): # also want FIFO join.
    wrap_target()
    label("loop")
    pull()
    out(pins, 7) [2]
    out(pins, 7) [2]
    out(pins, 7) [2]
    out(pins, 7)
    jmp("loop")
    wrap()
    
packed_data = array.array('I', [0 for _ in range(30)])
for i in range(30):
    outval = 0
    for j in range(4):
        value = (i*4)+j
        outval = outval | (value << j*7)
    packed_data[i] = outval
    
sm = rp2.StateMachine(0, seven_bit_dac, freq=50000, out_base=Pin(0))
sm.active(1)

while True:
    sm.put(packed_data,0)
    
