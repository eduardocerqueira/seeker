#date: 2024-05-01T16:40:17Z
#url: https://api.github.com/gists/e084a4f7fb018e2c4af3e5934e33e046
#owner: https://api.github.com/users/jepler

import displayio
displayio.release_displays()

import audiobusio
import audiocore
try:
    import audiomp3
except:
    audiomp3 = None
import board
import array
import time
import math
try:
    from ulab import numpy as np
except:
    np = None

TEST = 'mp3'

if TEST == 'wave':
    sample = audiocore.WaveFile(open("/sample.wav", "rb"))
elif TEST == 'mp3':
    sample = audiomp3.MP3Decoder("/sample.mp3")
else:
    from math import sin, pi
    SCALE = 3200
    OFFSET = 0
    sample_rate = 8800
    length = sample_rate // 440
    sine_wave = array.array('h', [int(sin(2 * pi * i / length) * SCALE + OFFSET) for i in range(length)])
    sample = audiocore.RawSample(sine_wave, sample_rate=sample_rate)
    print(f"{len(sine_wave)=}")

i2s = audiobusio.I2SOut(bit_clock=board.MOSI, word_select=board.SCK, data=board.MISO)
#i2s = audiobusio.I2SOut(bit_clock=board.A1, word_select=board.A0, data=board.A2, left_justified=True)
while True:
    print(".")
    if isinstance(sample, audiocore.RawSample):
        i2s.play(sample, loop=True)
        time.sleep(1)
    else:
        i2s.play(sample, loop=False)
        time.sleep(1)
        while i2s.playing:
            time.sleep(1)
    print()
    i2s.stop()
    time.sleep(.2)
