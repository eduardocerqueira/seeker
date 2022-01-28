#date: 2022-01-28T17:01:31Z
#url: https://api.github.com/gists/fff32409b3d8fb3f7d4cc9860185cb6b
#owner: https://api.github.com/users/jepler

# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2022 Jeff Epler for Adafruit Industries
#
# SPDX-License-Identifier: Unlicense

# On an Adafruit Feather M4 with Floppy Featherwing, do some track-to-track seeking.

import board
import array
import adafruit_floppy
import adafruit_ticks
import wait_serial

import ulab.numpy as np
import adafruit_pioasm
import rp2pio

D24 = getattr(board, 'D24') or getattr(board, 'A4')
D25 = getattr(board, 'D25') or getattr(board, 'A5')

indexpin=board.A1
rddatapin=board.D9

program = adafruit_pioasm.Program("""
.side_set 1
; Count flux pulses and watch for index pin
; flux input is the 'jmp pin'.  index is "pin zero".

; Counts are in units 3 / F_pio, so e.g., at 30MHz 1 count = 0.1us

; Count down while waiting for the counter to go HIGH
; The only counting is down, so C code will just have to negate the count!
; Each 'wait one' loop takes 3 instruction-times
wait_one:
    jmp x--, wait_one_next  side 1 ; acts as a non-conditional decrement of x
wait_one_next:
    jmp pin wait_zero
    jmp wait_one

; Each 'wait zero' loop takes 3 instruction-times, needing one instruction delay
; (it has to match the 'wait one' timing exactly)
wait_zero:
    jmp x--, wait_zero_next side 1 ; acts as a non-conditional decrement of x
wait_zero_next:
    jmp pin wait_zero [1]

; Top bit is index status, bottom 15 bits are inverse of counts
; Combined FIFO gives 16 entries (8 32-bit entries) so with the 
; smallest plausible pulse of 2us there are 250 CPU cycles available @125MHz
    in pins, 1
    in x, 15

; that's not working yet, just give 16 bits of 'x'
;    in x, 16
    jmp x--, wait_one
""")
#program = adafruit_pioasm.Program("""
#    jmp x--, next
#    nop
#next:
#    in pins, 1
#    in x, 15
#""")

print("kwargs", program.pio_kwargs)
print(" ".join("%04x" % instr for instr in program.assembled))

frequency = 125_000_000 // 3

buf = np.zeros(35000, dtype=np.uint16)
#raise SystemExit()

def test():
    with rp2pio.StateMachine(
        program.assembled,
        frequency=frequency,
        first_in_pin = indexpin,
        first_sideset_pin = board.TX,
        pull_in_pin_up = 1,
        jmp_pin = rddatapin,
        push_threshold = 32,
        auto_push = True,
        #**program.pio_kwargs
    ) as sm:

        f_pulses = sm.frequency / 3
        print(f"sm.frequency={sm.frequency}")
        print(f"f_pulses={f_pulses/1e6}MHz")
        print(f"1 count = {1e6/f_pulses}us")

        sm.readinto(memoryview(buf).cast('L'))
        print(min(buf), max(buf))

    bins = [0] * 400

    o = buf[0]
    print(list(buf[:30]))
    for i in range(0, len(buf)-1):
        n = buf[i]
        diff = o - n
        if diff < 0: diff += 65536

        if (not (n & 1) and (o & 1)): print(f"falling index @ {i}")
        diff //= 2

        if diff >= 400:
            print("OOR", diff, n, o)
        else:
            bins[diff] += 1
        o = n

    o = -1
    for i, b in enumerate(bins):
        if b <= 1:
            continue
        if i != o+1:
            print("---")
        o = i
        print(f"{i/f_pulses*1e6:7.2} [{i:4}] {b:5}")

def test_loopy():
    import pwmio
    for frequency in 500_000, 333_333, 250_000:
        print("----")
        print(f"loopback frequency test {frequency} {1e6/frequency}us")
        print("----")
        for duty in 8000, 32000, 50000:
            with pwmio.PWMOut(board.D4, frequency=frequency) as p:
                p.duty_cycle = duty
                test()

def test_floppy():
    print("real floppy test")
    floppy = adafruit_floppy.MFMFloppy(
        densitypin=board.A0,
        selectpin=board.A2,
        motorpin=board.A3,
        directionpin=D24,
        steppin=D25,
        track0pin=board.D11,
        protectpin=board.D10,
        sidepin=board.D6,
        readypin=board.D5,
        indexpin=indexpin,
        rddatapin=rddatapin,
    )
    floppy.selected = True
    floppy.spin = True
    floppy.track = 8
    floppy.track = 1
    floppy.side = 0

    import time

#    print("flux readinto")
#    buf = bytearray(512)
#    n_read = floppy.flux_readinto(buf)
#    print(n_read)
#    print(buf)


    floppy._rddata.deinit()
    floppy._index.deinit()

    for i in range(8):
        test()

#test_loopy()
test_floppy()
