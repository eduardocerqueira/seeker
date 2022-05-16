#date: 2022-05-16T16:49:47Z
#url: https://api.github.com/gists/e425aca510a656ad1bfbae5ed3e0aa9b
#owner: https://api.github.com/users/Sucareto

from time import sleep
keycode = {
    "KEY_1": '\x1e',  # Keyboard 1 and !
    "KEY_2": '\x1f',  # Keyboard 2 and @
    "KEY_3": '\x20',  # Keyboard 3 and #
    "KEY_4": '\x21',  # Keyboard 4 and $
    "KEY_5": '\x22',  # Keyboard 5 and %
    "KEY_6": '\x23',  # Keyboard 6 and ^
    "KEY_7": '\x24',  # Keyboard 7 and &
    "KEY_8": '\x25',  # Keyboard 8 and *
    "KEY_9": '\x26',  # Keyboard 9 and (
    "KEY_0": '\x27',  # Keyboard 0 and )

    "KEY_Q": '\x14',  # Keyboard q and Q
    "KEY_W": '\x1a',  # Keyboard w and W
    "KEY_E": '\x08',  # Keyboard e and E
    "KEY_R": '\x15',  # Keyboard r and R
    "KEY_T": '\x17',  # Keyboard t and T
    "KEY_Y": '\x1c',  # Keyboard y and Y
    "KEY_U": '\x18',  # Keyboard u and U
    "KEY_I": '\x0c',  # Keyboard i and I
    "KEY_O": '\x12',  # Keyboard o and O
    "KEY_P": '\x13',  # Keyboard p and P

    "KEY_A": '\x04',  # Keyboard a and A
    "KEY_S": '\x16',  # Keyboard s and S
    "KEY_D": '\x07',  # Keyboard d and D
    "KEY_F": '\x09',  # Keyboard f and F
    "KEY_G": '\x0a',  # Keyboard g and G
    "KEY_H": '\x0b',  # Keyboard h and H
    "KEY_J": '\x0d',  # Keyboard j and J
    "KEY_K": '\x0e',  # Keyboard k and K
    "KEY_L": '\x0f',  # Keyboard l and L

    "KEY_Z": '\x1d',  # Keyboard z and Z
    "KEY_X": '\x1b',  # Keyboard x and X
    "KEY_C": '\x06',  # Keyboard c and C
    "KEY_V": '\x19',  # Keyboard v and V
    "KEY_B": '\x05',  # Keyboard b and B
    "KEY_N": '\x11',  # Keyboard n and N
    "KEY_M": '\x10',  # Keyboard m and M
}

shift_keycode = {
    "KEY_`": '\x00\x00\x35',  # Keyboard ` and ~
    "KEY_-": '\x00\x00\x2d',  # Keyboard - and _
    "KEY_=": '\x00\x00\x2e',  # Keyboard = and +

    "KEY_~": '\x02\x00\x35',  # Keyboard ` and ~
    "KEY__": '\x02\x00\x2d',  # Keyboard - and _
    "KEY_+": '\x02\x00\x2e',  # Keyboard = and +

    "KEY_!": '\x02\x00\x1e',  # Keyboard 1 and !
    "KEY_@": '\x02\x00\x1f',  # Keyboard 2 and @
    "KEY_#": '\x02\x00\x20',  # Keyboard 3 and #
    "KEY_$": '\x02\x00\x21',  # Keyboard 4 and $
    "KEY_%": '\x02\x00\x22',  # Keyboard 5 and %
    "KEY_^": '\x02\x00\x23',  # Keyboard 6 and ^
    "KEY_&": '\x02\x00\x24',  # Keyboard 7 and &
    "KEY_*": '\x02\x00\x25',  # Keyboard 8 and *
    "KEY_(": '\x02\x00\x26',  # Keyboard 9 and (
    "KEY_)": '\x02\x00\x27',  # Keyboard 0 and )

    "KEY_[": '\x00\x00\x2f',  # Keyboard [ and {
    "KEY_]": '\x00\x00\x30',  # Keyboard ] and }
    "KEY_\\": '\x00\x00\x31',  # Keyboard \ and |
    "KEY_;": '\x00\x00\x33',  # Keyboard ; and :
    "KEY_'": '\x00\x00\x34',  # Keyboard ' and "
    "KEY_,": '\x00\x00\x36',  # Keyboard ', and <
    "KEY_.": '\x00\x00\x37',  # Keyboard . and >
    "KEY_/": '\x00\x00\x38',  # Keyboard / and ?

    "KEY_{": '\x02\x00\x2f',  # Keyboard [ and {
    "KEY_}": '\x02\x00\x30',  # Keyboard ] and }
    "KEY_|": '\x02\x00\x31',  # Keyboard \ and |
    "KEY_:": '\x02\x00\x33',  # Keyboard ; and :
    'KEY_"': '\x02\x00\x34',  # Keyboard ' and "
    "KEY_<": '\x02\x00\x36',  # Keyboard ', and <
    "KEY_>": '\x02\x00\x37',  # Keyboard . and >
    "KEY_?": '\x02\x00\x38',  # Keyboard / and ?
    "KEY_ ": '\x02\x00\x2c',  # Keyboard Spacebar
}

with open("/dev/hidg0", "rb+", buffering=0) as f:
    while True:
        line = input()
        for i in line:
            data = "\x00"
            if i.isalnum():
                if i.isupper():
                    data = "\x02"
                data += "\x00"
                data += keycode["KEY_"+i.upper()]
            else:
                data = shift_keycode["KEY_"+i]
            data += '\x00\x00\x00\x00\x00'
            f.write(data.encode())
            f.write('\x00\x00\x00\x00\x00\x00\x00\x00'.encode())
            sleep(0.01)
        f.write('\x00\x00\x28\x00\x00\x00\x00\x00'.encode())
        f.write('\x00\x00\x00\x00\x00\x00\x00\x00'.encode())
