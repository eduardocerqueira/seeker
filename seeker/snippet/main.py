#date: 2022-10-27T17:07:15Z
#url: https://api.github.com/gists/d3b472d4f8c7935800c43e9938a79a9d
#owner: https://api.github.com/users/sbkeebs

# Microdox v2 

import board
from kmk.kmk_keyboard import KMKKeyboard
from kmk.scanners import DiodeOrientation

from kmk.keys import KC
from kmk.modules.layers import Layers
from kmk.modules.split import Split, SplitSide, SplitType


keyboard = KMKKeyboard()
keyboard.col_pins = (board.GP04, board.GP06, board.GP20, board.GP26, board.GP27)
keyboard.row_pins = (board.GP29, board.RX, board.GP05, board.GP22)
keyboard.diode_orientation = DiodeOrientation.COL2ROW
keyboard.data_pin = board.TX
keyboard.coord_mapping = [
     0,  1,  2,  3,  4,  20, 21, 22, 23, 24,
     5,  6,  7,  8,  9,  25, 26, 27, 28, 29,
    10, 11, 12, 13, 14,  30, 31, 32, 33, 34,
            17, 18, 19,  35, 36, 37,
]

split_side = SplitSide.LEFT # change to RIGHT for right side
split = Split(
    data_pin=board.TX, 
    split_side=split_side,
)

layers_ext = Layers()

keyboard.modules = [layers_ext, split]

# Cleaner key names
_______ = KC.TRNS
XXXXXXX = KC.NO

LOWER = KC.MO(2)
RAISE = KC.MO(1)

keyboard.keymap = [
    [  #QWERTY
        KC.Q,    KC.W,    KC.E,    KC.R,    KC.T,                         KC.Y,    KC.U,    KC.I,    KC.O,   KC.P,\
        KC.A,    KC.S,    KC.D,    KC.F,    KC.G,                         KC.H,    KC.J,    KC.K,    KC.L, KC.SCLN,\
        KC.Z,    KC.X,    KC.C,    KC.V,    KC.B,                         KC.N,    KC.M, KC.COMM,  KC.DOT, KC.SLSH,\
                                    KC.LCTL,   LOWER,  KC.SPC,     KC.BSPC,    RAISE,  KC.ENT,
    ],
    [  #RAISE
        KC.N1,   KC.N2,   KC.N3,   KC.N4,   KC.N5,                        KC.N6,   KC.N7,   KC.N8,   KC.N9,   KC.N0,\
        KC.TAB,  KC.LEFT, KC.DOWN, KC.UP,   KC.RGHT,                      XXXXXXX, KC.MINS, KC.EQL,  KC.LBRC, KC.RBRC,\
        KC.LCTL, KC.GRV,  KC.LGUI, KC.LALT, XXXXXXX,                      XXXXXXX, XXXXXXX, XXXXXXX, KC.BSLS, KC.QUOT,\
                                    XXXXXXX, XXXXXXX, XXXXXXX,      XXXXXXX, XXXXXXX, XXXXXXX,
    ],
    [  #LOWER
        KC.EXLM, KC.AT,   KC.HASH, KC.DLR,  KC.PERC,      KC.CIRC, KC.AMPR, KC.ASTR, KC.LPRN, KC.RPRN,\
        KC.ESC,  XXXXXXX, XXXXXXX, XXXXXXX, XXXXXXX,      XXXXXXX, KC.UNDS, KC.PLUS, KC.LCBR, KC.RCBR,\
        KC.CAPS, KC.TILD, XXXXXXX, XXXXXXX, XXXXXXX,      XXXXXXX, XXXXXXX, XXXXXXX, KC.PIPE,  KC.DQT,\
                            XXXXXXX, XXXXXXX, XXXXXXX,      KC.ENT,  XXXXXXX, KC.DEL
    ]
]

if __name__ == '__main__':
    keyboard.go()