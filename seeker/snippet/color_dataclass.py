#date: 2026-03-11T17:27:32Z
#url: https://api.github.com/gists/2ce96de2ffb4fe237680b6a82235ab69
#owner: https://api.github.com/users/pythonhacker

from dataclasses import dataclass

@dataclass(frozen=True)
class ColorData:
    r: int = 0
    g: int = 0
    b: int = 0

    def __post_init__(self):
        for v in (self.r, self.g, self.b):
            if not 0 <= v <= 255:
                raise ValueError("RGB must be 0..255")