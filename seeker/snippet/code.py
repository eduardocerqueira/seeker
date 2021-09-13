#date: 2021-09-13T17:03:36Z
#url: https://api.github.com/gists/0e426cc9d150c078f7f0b276afa6ff57
#owner: https://api.github.com/users/anecdata

import sys
import board
import displayio
import terminalio

display = board.DISPLAY  # or equivalent external display library

splash = displayio.Group()

fontx, fonty = terminalio.FONT.get_bounding_box()
term_palette = displayio.Palette(2)
term_palette[0] = 0x000000
term_palette[1] = 0xffffff
logbox = displayio.TileGrid(terminalio.FONT.bitmap,
                            x=0,
                            y=0,
                            width=display.width // fontx,
                            height=display.height // fonty,
                            tile_width=fontx,
                            tile_height=fonty,
                            pixel_shader=term_palette)
splash.append(logbox)
logterm = terminalio.Terminal(logbox, terminalio.FONT)

display.show(splash)

print("Hello Serial!", file=sys.stdout)  # serial console
print("\r\nHello displayio!", file=logterm, end="")  # displayio
