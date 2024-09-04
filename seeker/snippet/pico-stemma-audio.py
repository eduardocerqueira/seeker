#date: 2024-09-04T16:59:48Z
#url: https://api.github.com/gists/088e038338c79e5dfbd23584a2e5544a
#owner: https://api.github.com/users/FalcoG

# SPDX-FileCopyrightText: 2023 Kattni Rembor for Adafruit Industries
#
# SPDX-License-Identifier: MIT

"""
CircuitPython PWM Audio Short Tone Tune Demo

Plays a five-note tune on a loop.
"""
import time
import array
import math
import random
import board
import asyncio
from digitalio import DigitalInOut, Direction, Pull
from audiocore import RawSample, WaveFile
from audiopwmio import PWMAudioOut as AudioOut

print("Hello World!")

audio = AudioOut(board.GP0)

# Increase this to increase the volume of the tone.
tone_volume = 0.4
# The tones are provided as a frequency in Hz. You can change the current tones or
# add your own to make a new tune. Follow the format with commas between values.
tone_frequency = [784, 880, 698, 349, 523]

btn = DigitalInOut(board.GP15)
btn.direction = Direction.INPUT
btn.pull = Pull.UP

def audio_tone(frequency):
    # Compute the sine wave for the current frequency.
    length = 8000 // frequency
    sine_wave = array.array("H", [0] * length)
    for index in range(length):
        sine_wave[index] = int((1 + math.sin(math.pi * 2 * index / length))
                               * tone_volume * (2 ** 15 - 1))

    sine_wave_sample = RawSample(sine_wave)

    # Play the current frequency.
    audio.play(sine_wave_sample, loop=True)


def audio_tunes():
    for frequency in tone_frequency:
        audio_tone(frequency)
        time.sleep(0.25)
        audio.stop()

async def button(event):
    while True:
        await asyncio.sleep(0)
        cur_state = not btn.value # invert to make True for keydown
        if event.is_set() != cur_state:
            if cur_state:
                print("BTN is down")
                event.set()
            else:
                print("BTN is up")
                event.clear()

async def play(event):
    last_state = event.is_set()
    while True:
        await asyncio.sleep(0)
        if event.is_set() != last_state:
            last_state = event.is_set()

            if last_state:
                #freq = random.choice(tone_frequency)
                freq = random.randint(340, 900)
                print(freq)
                #audio_tunes()
                audio_tone(freq)
                print("pressing")
            else:
                print("STOP NOW")
                audio.stop()

button_active = asyncio.Event()

async def main():
    btn = asyncio.create_task(button(button_active))
    playing = asyncio.create_task(play(button_active))
    await btn
    await playing


asyncio.run(main())
