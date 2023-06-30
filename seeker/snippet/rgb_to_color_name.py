#date: 2023-06-30T16:45:58Z
#url: https://api.github.com/gists/243e213a89582cfdac5af2904cda00f2
#owner: https://api.github.com/users/hatarist

from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color


HUE_LIMITS_FOR_SAT = (
    (8,  0,  0, 44,  0,  0,  0, 63,  0,   0, 122,   0, 134,   0,   0,   0,   0, 166, 176, 241,   0, 256,   0),  # Sat: 20-75
    (0, 10,  0, 32, 46,  0,  0,  0, 61,   0, 106,   0, 136, 144,   0,   0,   0, 158, 166, 241,   0,   0, 256),  # Sat: 75-115
    (0,  8,  0,  0, 39, 46,  0,  0,  0,  71, 120,   0, 131, 144,   0,   0, 163,   0, 177, 211, 249,   0, 256),  # Sat: 115-150
    (0, 11, 26,  0,  0, 38, 45,  0,  0,  56, 100, 121, 129,   0, 140,   0, 180,   0,   0, 224, 241,   0, 256),  # Sat: 150-240
    (0, 13, 27,  0,  0, 36, 45,  0,  0,  59, 118,   0, 127, 136, 142,   0, 185,   0,   0, 216, 239,   0, 256),  # Sat: 240-255
)

LUM_LIMITS_FOR_HUE = (
    (130, 100, 115, 100, 100, 100, 110,  75, 100,  90, 100, 100, 100, 100,  80, 100, 100, 100, 100, 100, 100, 100, 100),
    (170, 170, 170, 155, 170, 170, 170, 170, 170, 115, 170, 170, 170, 170, 170, 170, 170, 170, 150, 150, 170, 140, 165)
)


COLOR_NAMES = (
    # light
    ("coral", "rose", "lightorange", "tan", "tan", "lightyellow", "lightyellow", "tan", "lightgreen", "lime", "lightgreen", "lightgreen", "aqua", "skyblue", "lightturquoise", "paleblue", "lightblue", "iceblue", "periwinkle", "lavender", "pink", "tan", "rose"),
    
    # mid
    ("coral", "red", "orange", "brown", "tan", "gold", "yellow", "olivegreen", "olivegreen", "green", "green", "brightgreen", "teal", "aqua", "turquoise", "paleblue", "blue", "bluegray", "indigo", "purple", "pink", "brown", "red"),
    
    # dark
    ("brown", "darkred", "brown", "brown", "brown", "darkyellow", "darkyellow", "brown", "darkgreen", "darkgreen", "darkgreen", "darkgreen", "darkteal", "darkteal", "darkteal", "darkblue", "darkblue", "bluegray", "indigo", "darkpurple", "plum", "brown", "darkred"),
)


def rgb_to_hsl(r, g, b):
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    hsl = convert_color(rgb, HSLColor, target_illuminant='d50')
    return hsl.get_value_tuple()


def get_color_name(rgb):
    hsl = rgb_to_hsl(*rgb)
    hue, sat, lum = hsl

    hue = (0 if hue == 0 else hue / 360) * 255  # using normalization to 0-255 instead of 0-360°
    sat = sat * 255
    lum = lum * 255

    if lum > 240:
        return "white"
    elif lum < 20:
        return "black"

    if sat <= 20:
        if (lum > 170):
            return "lightgray"
        elif (lum > 100):
            return "gray"
        else:
            return "darkgray"

    if (sat > 20 and sat <= 75):
        pHueLimits = HUE_LIMITS_FOR_SAT[0]
    elif (sat > 75 and sat <= 115):
        pHueLimits = HUE_LIMITS_FOR_SAT[1]
    elif (sat > 115 and sat <= 150):
        pHueLimits = HUE_LIMITS_FOR_SAT[2]
    elif (sat > 150 and sat <= 240):
        pHueLimits = HUE_LIMITS_FOR_SAT[3]
    else:
        pHueLimits = HUE_LIMITS_FOR_SAT[4]

    colorIndex = -1
    for i in range(0, len(COLOR_NAMES[1])):
        if hue < pHueLimits[i]:
            colorIndex = i
            break

    if colorIndex != -1:
        if (lum > LUM_LIMITS_FOR_HUE[1][colorIndex]):
            return COLOR_NAMES[0][colorIndex]  # light
        elif (lum < LUM_LIMITS_FOR_HUE[0][colorIndex]):
            return COLOR_NAMES[2][colorIndex]  # dark
        else:
            return COLOR_NAMES[1][colorIndex]  # mid

    return ''
