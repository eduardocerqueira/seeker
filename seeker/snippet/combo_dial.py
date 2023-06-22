#date: 2023-06-22T16:36:28Z
#url: https://api.github.com/gists/1b6b7922f06bcc2776a5152428a6fb1e
#owner: https://api.github.com/users/jedgarpark

# Rotary dial PyPortal combo display
# displays incremental position and full rotations
# tap the screen to zero it (first put dial at "0")

import time
import board
import rotaryio
import adafruit_touchscreen
import displayio
import terminalio
from adafruit_display_text import label


encoder = rotaryio.IncrementalEncoder (board.SCL, board.SDA)
max_number = 100

display = board.DISPLAY
main_display_group = displayio.Group()
display.show(main_display_group)
background = displayio.Bitmap(display.width, display.height, 1)

color_pal = displayio.Palette(3)
color_pal[0] = 0x302000  # dark
color_pal[1] = 0xc08000  # mid
color_pal[2] = 0xf09000  # bright

#Put background into main group, using palette to map palette ids to colors
main_display_group.append(displayio.TileGrid(background, pixel_shader=color_pal))

background.fill(0)

text_area_dial_num = label.Label(terminalio.FONT, text="0", x=(display.width//2-40),  # large number
                        y=(display.height//2), scale = 7,
                        background_color=color_pal[0], color=color_pal[2] )
main_display_group.append(text_area_dial_num)

text_area_rev = label.Label(terminalio.FONT, text="0", x=(display.width//2-110),  # small number
                        y=(display.height//2), scale = 4,
                        background_color=color_pal[0], color=color_pal[1] )
main_display_group.append(text_area_rev)

ts = adafruit_touchscreen.Touchscreen(
                                        board.TOUCH_XL, board.TOUCH_XR,
                                        board.TOUCH_YD, board.TOUCH_YU,
)

last_position = 0  # for position state


while True:
    touch = ts.touch_point  # check for touch
    if touch:
        encoder.position = 0  # reset value
        text_area_dial_num.text = ("0")
        text_area_rev.text = ("0")

    position = encoder.position  # get encoder position
    if position != last_position:  # if it has changed
        dial_number = position % max_number  # modulo of max value
        revolution = position // max_number  # integer divide max_number
        delta = position - last_position  # how big a change
        if delta > 0:  # it increased
            for _ in range(delta):
                print("rev:", revolution, ", dial number:", dial_number)
                text_area_dial_num.text=(str(dial_number))
                text_area_rev.text=(str(revolution))

        if delta < 0:  # it decreased
            for _ in range(-delta):
                print("rev:", revolution, ", dial number:", dial_number)
                text_area_dial_num.text=(str(dial_number))
                text_area_rev.text=(str(revolution))

        last_position = position  # save new state
