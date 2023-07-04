#date: 2023-07-04T16:41:48Z
#url: https://api.github.com/gists/01870c50fb6eb2820c570f1cf4323823
#owner: https://api.github.com/users/todbot

# originally from DanG
import board
import analogio
import audiopwmio
import keypad
import synthio
from ulab.numpy import np

# Define the switch and potentiometer pins
switch_pin = board.A1
potentiometer_pin = board.A0
amplifier_pin = board.A2

# Initialize the switch and potentiometer
# use keypad to debounce the switch for us
switch = keypad.Keys( (switch_pin,), value_when_pressed=False)

potentiometer = analogio.AnalogIn(potentiometer_pin)

# create audio setup
audio = audiopwmio.PWMAudioOut(amplifier_pin)
synth = synthio.Synthesizer(sample_rate=28000)
audio.play(synth)

# make a sinewave to use as oscillator instead of default square
wave_sine = np.array(np.sin(np.linspace(0, 2*np.pi, 512, endpoint=False)) * 32000, dtype=np.int16)

# make an envelope with nicer attack and longer release time and max levels
amp_env = synthio.Envelope(attack_time=0.1, release_time=1.5)

# Main loop
while True:
    event = switch.events.get()
    if event.pressed:
        # Read the potentiometer value
        pot_value = potentiometer.value

        # Start a note playing forever at 640 Hz, with loudness based on pot
        note = synthio.Note( frequency=640,
                             waveform=wave_sine,
                             envelope=amp_env,
                             amplitude = pot_value / 65535
                            )
        synth.press(note)

    if event.released:
        # Stop playing the note
        if note:
            synth.release(note)