#date: 2024-08-27T16:47:35Z
#url: https://api.github.com/gists/abf1e2bfe7149309acc3606cc70619a7
#owner: https://api.github.com/users/ikeji

print("Starting")

from kmk.keys import KC
from kmk.kmk_keyboard import KMKKeyboard as _KMKKeyboard
from kmk.modules.layers import Layers
from kmk.modules.sticky_keys import StickyKeys
from kmk.scanners.keypad import KeysScanner
import board

_PINS = [
    board.GP2, board.GP6, board.GP10, board.GP13,
    board.GP3, board.GP7, board.GP11, board.GP12,
]

class KMKKeyboard(_KMKKeyboard):
    coord_mapping = [
        0, 1, 2, 3,
        4, 5, 6, 7,
    ]

    def __init__(self):
        self.matrix = KeysScanner(_PINS)

keyboard = KMKKeyboard()

sticky_keys = StickyKeys()

keyboard.modules.append(Layers())
keyboard.modules.append(sticky_keys)

keyboard.keymap = [
    [
        KC.SK(KC.LCTL), KC.LT(1, KC.B), KC.C, KC.D,
        KC.E, KC.F, KC.G, KC.H,
    ],
    [
        KC.I, KC.TRNS, KC.K, KC.L,
        KC.M, KC.N, KC.O, KC.P,
    ],
]

if __name__ == '__main__':
    keyboard.go()
