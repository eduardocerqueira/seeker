#date: 2022-05-12T16:56:41Z
#url: https://api.github.com/gists/c0da9b2b81544ecf21220ed2a1321dd7
#owner: https://api.github.com/users/maditnerd

import board
import displayio
import terminalio
import adafruit_displayio_ssd1306
from adafruit_display_text import label
import time
# Create I2C bus as normal
displayio.release_displays()

def tca_select(channel):
    while not i2c.try_lock():
        pass
    i2c.writeto(0x70, bytearray([1 << channel]))
    i2c.unlock()

i2c = board.I2C()  # uses board.SCL and board.SDA
n = 0
while True:
    for i in range(0,8):
        tca_select(i)
        display_bus = displayio.I2CDisplay(i2c, device_address=0x3C)
        display = adafruit_displayio_ssd1306.SSD1306(display_bus, width=100, height=42)
        print("TCA Channel: " + str(i))
        text = str(i + n)
        text_area = label.Label(terminalio.FONT, text=text, scale=5, color=0xFFFF00, x=40, y=16)
        display.show(text_area)
        time.sleep(0.05)
        displayio.release_displays()
    n = n + 1