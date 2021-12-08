#date: 2021-12-08T17:01:15Z
#url: https://api.github.com/gists/01aed051251476c4bd6daa4b076eb23a
#owner: https://api.github.com/users/Integralist

# https://mdk.fr/blog/how-apt-does-its-fancy-progress-bar.html
#
# For the record here's what's used (\033 is ESC):
#
# `ESC 7`           is DECSC   (Save Cursor)
# `ESC 8`           is DECRC   (Restore Cursor)
# `ESC [ Pn ; Pn r` is DECSTBM (Set Top and Bottom Margins)
# `ESC [ Pn A`      is CUU     (Cursor Up)
# `ESC [ Pn ; Pn f` is HVP     (Horizontal and Vertical Position)
# `ESC [ Ps K`      is EL      (Erase In Line)

import os
import time
from datetime import datetime

columns, lines = os.get_terminal_size()


def write(s):
    print(s, end="")
    time.sleep(1)


write("\n")                  # Ensure the last line is available.
write("\0337")               # Save cursor position
write(f"\033[0;{lines-1}r")  # Reserve the bottom line
write("\0338")               # Restore the cursor position
write("\033[1A")             # Move up one line

try:
    for i in range(250):
        time.sleep(0.2)
        write(f"Hello {i}")
        write("\0337")                     # Save cursor position
        write(f"\033[{lines};0f")          # Move cursor to the bottom margin
        write(datetime.now().isoformat())  # Write the date
        write("\0338")                     # Restore cursor position
        write("\n")
except KeyboardInterrupt:
    pass
finally:
    write("\0337")             # Save cursor position
    write(f"\033[0;{lines}r")  # Drop margin reservation
    write(f"\033[{lines};0f")  # Move the cursor to the bottom line
    write("\033[0K")           # Clean that line
    write("\0338")             # Restore cursor position