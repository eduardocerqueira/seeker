#date: 2025-08-01T16:58:45Z
#url: https://api.github.com/gists/dccaeda7ab29ab69286fd77d84ef4400
#owner: https://api.github.com/users/gallaugher

```# text-and-displayio.py
import board, busio, time, displayio, pwmio, terminalio, fourwire
from adafruit_display_text.label import Label
from adafruit_display_text.bitmap_label import Label as BitmapLabel
from adafruit_bitmap_font import bitmap_font
import adafruit_ili9341

# --- Display Setup ---
displayio.release_displays()

# SPI bus for display
spi = busio.SPI(clock=board.GP18, MOSI=board.GP19)

# Display control pins
tft_cs = board.GP20
tft_dc = board.GP21
tft_reset = board.GP15

# Display bus
display_bus = fourwire.FourWire(
    spi,
    command=tft_dc,
    chip_select=tft_cs,
    reset=tft_reset
)

# Initialize display in landscape mode
display = adafruit_ili9341.ILI9341(
    display_bus,
    width=320,
    height=240,
    rotation=0,  # Landscape mode
    backlight_pin=None  # Board has a backlight but we'll handle it w/pwm so we can dim.
)

# PWM backlight - note we ignore the backlight pin.
# to change backlight, set: backlight.duty_cycle = to range 0-65535
backlight = pwmio.PWMOut(board.GP22, frequency=5000, duty_cycle=65535)

group = displayio.Group()  # creates an empty group
display.root_group = group  # sets this group as our primary or "root" group

label_font = bitmap_font.load_font("/fonts/helvB18.bdf")

# Create a bottom_left label
bottom_left = Label(
    label_font,
    text="Bottom Left",
    anchor_point=(0, 1),  # Anchor left, bottom
    anchored_position=(0, display.height - 1),  # position at left, bottom
    color=(255, 255, 255),
)
group.append(bottom_left)

bottom_right = BitmapLabel(
    label_font,
    text="Bottom Right",
    anchor_point=(1, 1),  # Anchor right, bottom
    anchored_position=(display.width - 1, display.height - 1),
    base_alignment=False,
    color=(0, 255, 0),
)
group.append(bottom_right)

# bottom_right = BitmapLabel(
#    label_font,
#    text="Bottom Right",
#    anchor_point=(1, 1), # Anchor right, bottom
#    color=(0, 255, 0),
# )
# y_offset = bottom_right.bounding_box[3]
# bottom_right.anchored_position = (display.width-1, display.height-1-y_offset)
# group.append(bottom_right) # positions higher than expected descent

# Be sure to continue with an infinite loop or the code will stop & the display will clear.
while True:
    pass
```