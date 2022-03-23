#date: 2022-03-23T17:09:17Z
#url: https://api.github.com/gists/b84fa5eb8b4807141372487ec4333303
#owner: https://api.github.com/users/farooqkz

# By Farooq Karimi Zadeh under CC0.

import wave
import struct
from dataclasses import dataclass
from typing import List


@dataclass
class SoundClip16b:
    data: List["int"]

    def __len__(self):
        return len(self.data)

    def inverted(self):
        inv = list()
        for d in self.data:
            inv.append(~d)
        return SoundClip16b(data=inv)

    def to_bytes(self):
        databyte = b""
        for d in self.data:
            databyte += struct.pack("<h", d)
        return databyte

    @staticmethod
    def from_bytes(bytedata):
        return SoundClip16b(
            data=[x[0] for x in struct.iter_unpack("<h", bytedata)]
        )


w_noise = wave.open("noise.wav")
w_antinoise = wave.open("antinoise.wav", "w")
clip = SoundClip16b.from_bytes(w_noise.readframes(-1))
inverted_clip = clip.inverted()
w_antinoise.setnchannels(1)
w_antinoise.setsampwidth(2)
w_antinoise.setframerate(8000)
w_antinoise.writeframes(inverted_clip.to_bytes())
w_antinoise.close()