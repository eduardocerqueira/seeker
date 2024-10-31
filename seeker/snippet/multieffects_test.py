#date: 2024-10-31T16:59:19Z
#url: https://api.github.com/gists/4e48a96d8c789490e9f17e3f4f8124ca
#owner: https://api.github.com/users/dcooperdalrymple

# Audio signal path: Sample => HPF => Distortion => LPF => Delay => Mixer => Output

import board
import audiobusio
import audiocore
import audiodelays
import audiofilters
import audiomixer
import synthio

audio = audiobusio.I2SOut(bit_clock=board.GP0, word_select=board.GP1, data=board.GP2)

wave_file = open("StreetChicken.wav", "rb")
wave = audiocore.WaveFile(wave_file)

# Used to generate filter biquads
synth = synthio.Synthesizer(sample_rate=wave.sample_rate)

# Shared properties among each effect
properties = {
    "buffer_size": 1024,
    "channel_count": wave.channel_count,
    "sample_rate": wave.sample_rate,
    "bits_per_sample": wave.bits_per_sample,
}

mixer = audiomixer.Mixer(
    **properties,
    voice_count=1,
)
audio.play(mixer)

hpf = audiofilters.Filter(
    **properties,
    mix=1.0,
    filter=synth.high_pass_filter(200),
)

# Available in PR #9776 (https://github.com/adafruit/circuitpython/pull/9776)
distortion = audiofilters.Distortion(
    **properties,
    mix=1.0,
    pre_gain=-5.0,
    post_gain=-5.0,
    drive=0.75,
    mode=audiofilters.DistortionMode.LOFI,
)

lpf = audiofilters.Filter(
    **properties,
    mix=1.0,
    filter=synth.low_pass_filter(4000, 2.0),
)

delay = audiodelays.Echo(
    **properties,
    mix=0.25,
    max_delay_ms=100,
    delay_ms=100,
    decay=0.5,
)

hpf.play(wave, loop=True)
distortion.play(hpf)
lpf.play(distortion)
delay.play(lpf)
mixer.play(delay)

time.sleep(10)
delay.stop()
# BUG: Stopping other effects in the chain before delay causes a crash. `hpf.play(wave, loop=False)` has the same effect.
