#date: 2022-04-20T17:21:47Z
#url: https://api.github.com/gists/65dd4cb8f1dddc3b77837d95bf2b3be3
#owner: https://api.github.com/users/aaronmelton

# SPDX-FileCopyrightText: 2021 Phillip Burgess for Adafruit Industries
#
# SPDX-License-Identifier: MIT

from adafruit_hid.keycode import Keycode # REQUIRED if using Keycode.* values

app = {               # REQUIRED dict, must be named 'app'
    'name' : 'Meetings', # Application name
    'macros' : [      # List of button macros...
        # COLOR    LABEL    KEY SEQUENCE
        # 1st row ----------
        (0x000000, '', []),
        (0x000000, '-------[Zoom]-------', []),
        (0x000000, '', []),
        # 2nd row ----------
        (0x004dcf, 'Mute', [Keycode.COMMAND, Keycode.SHIFT, 'a']),
        (0xfccb00, 'Camera', [Keycode.COMMAND, Keycode.SHIFT, 'v']),
        (0xb80000, 'Quit', [Keycode.COMMAND, 'w', 1.0, Keycode.ENTER]),
        # 3rd row ----------
        (0x000000, '', []),
        (0x000000, '----[Google Meet]----', []),
        (0x000000, '', []),
        # 4th row ----------
        (0x004dcf, 'Mute', [Keycode.COMMAND, 'd']),
        (0xfccb00, 'Camera', [Keycode.COMMAND, 'e']),
        (0xb80000, 'Quit', [Keycode.COMMAND, 'w']),
        # Encoder button ---
        (0x000000, '', [])
    ]
}
