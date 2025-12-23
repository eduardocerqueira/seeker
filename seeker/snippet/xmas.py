#date: 2025-12-23T16:57:34Z
#url: https://api.github.com/gists/f28c7c6ca81af8678c1d778d14dc2230
#owner: https://api.github.com/users/jon2allen

#############################
# Created by Jon Allen (modified)
# December 2025
#############################
import curses
import time

def draw_box(win, y, x, height, width, title):
    win.attron(curses.color_pair(3))
    for i in range(width):
        win.addch(y, x + i, curses.ACS_HLINE)
        win.addch(y + height, x + i, curses.ACS_HLINE)
    for i in range(height):
        win.addch(y + i, x, curses.ACS_VLINE)
        win.addch(y + i, x + width, curses.ACS_VLINE)
    win.addch(y, x, curses.ACS_ULCORNER)
    win.addch(y, x + width, curses.ACS_URCORNER)
    win.addch(y + height, x, curses.ACS_LLCORNER)
    try:
        win.move(y + height, x + width)
        win.insch(curses.ACS_LRCORNER)
    except:
        pass
    win.addstr(y, x + 2, f" {title} ")
    win.attroff(curses.color_pair(3))

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLACK)

    # Christmas tree ASCII art
    christmas_art = [
        "    *            *            *            *          ",
        "   ***          ***          ***          ***         ",
        "  *****        *****        *****        *****        ",
        " *******      *******      *******      *******       ",
        "*********    *********    *********    *********      ",
        "   |||         |||          |||          |||          ",
        "   |||         |||          |||          |||          ",
        "               MERRY CHRISTMAS!                         "
    ]

    # Merry Christmas in different languages

    merry_christmas  = [
      "MERRY CHRISTMAS!",              # English
      "JOYEUX NOËL!",                 # French
      "FROHE WEIHNACHTEN!",           # German
      "圣诞快乐!",                      # Chinese (Mandarin)
      "С РОЖДЕСТВОМ!",               # Russian
      "عيد ميلاد مجيد!",  # Arabic
      "ⲠⲓⲬⲣⲓⲥⲧⲟⲥ ⲁⲩⲙⲁⲥϥ!",           # Coptic ("Christ is Born!")
      "ΚΑΛΑ ΧΡΙΣΤΟΥΓΕΝΝΑ!",           # Greek
      "FELICEM NATIVITATEM!",         # Latin (Vatican/Ecclesiastical style)
      "MUTLU NOELLER!"                # Turkish
    ]
    merry_christmas2 = [
        "MERRY CHRISTMAS!",
        "JOYEUX NOËL!",
        "FROHE WEIHNACHTEN!",
        "圣诞快乐!",  # Chinese
        "С РОЖДЕСТВОМ!",  # Russian
        "عيد ميلاد مجيد!",  # Arabic
        "ⲭⲣⲓⲥⲧⲟⲥ Ⲁⲃⲉⲛⲉ!",  # Coptic
        "ΚΑΛΑ ΧΡΙΣΤΟΥΓΕΝΝΑ!",  # Greek
        "FELIX NATALIS!",  # Latin
        "MUTLU NOELLER!"  # Turkish
    ]

    # Christmas colors (red, green, gold, blue, white)
    christmas_colors = [2, 1, 4, 7, 6]

    # State
    scroll_pos = 0
    scroll_speed = 0.1
    paused = False
    language_index = 0
    color_index = 0
    language_change_counter = 0

    while True:
        key = stdscr.getch()

        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('+'): scroll_speed = max(0.01, scroll_speed - 0.01)
        elif key == ord('-'): scroll_speed = min(0.5, scroll_speed + 0.01)

        # --- DRAWING ---
        stdscr.erase()

        # Draw the main box
        sh, sw = stdscr.getmaxyx()
        draw_box(stdscr, 1, 1, sh - 3, sw - 2, "MERRY CHRISTMAS SCROLLER")

        # Draw scrolling Christmas art
        for i, line in enumerate(christmas_art):
            if i + 4 < sh - 2:  # Make sure we don't go out of bounds
                # Calculate visible portion of the line
                visible_line = line[scroll_pos:] + line[:scroll_pos]
                visible_line = visible_line[:sw-4]  # Truncate to window width

                # Draw with alternating colors for festive effect
                color = curses.color_pair(1) if i % 2 == 0 else curses.color_pair(2)
                if i == 7:  # The "MERRY CHRISTMAS" line
                    # Update language and color periodically
                    if not paused:
                        language_change_counter += 1
                        if language_change_counter >= 20:
                            language_change_counter = 0
                            language_index = (language_index + 1) % len(merry_christmas)
                            color_index = (color_index + 1) % len(christmas_colors)
                    # Replace the text in the line
                    visible_line = merry_christmas[language_index].center(len(line))
                    color = curses.color_pair(christmas_colors[color_index]) | curses.A_BOLD
                stdscr.addstr(3 + i, 2, visible_line, color)

        # Draw some Christmas ornaments that scroll with the trees
        ornaments = [
            (5, 10),  # row 5, offset 10 from tree start
            (7, 15),  # row 7, offset 15 from tree start
            (4, 20),
            (8, 25),  # This was in the original text line, moved to row 8
            (6, 30),
            (3, 35),
            (9, 40),
        ]

        for i, (y, x_offset) in enumerate(ornaments):
            if y + 3 < sh - 2 and y != 7:  # Skip the text line (row 7)
                # Calculate ornament position based on scroll position
                ornament_x = (x_offset - scroll_pos) % (len(christmas_art[0]) - 1)
                if ornament_x < 0:
                    ornament_x += len(christmas_art[0]) - 1
                if ornament_x + 2 < sw - 2:
                    color = curses.color_pair(1 + (i % 5))  # Cycle through colors
                    stdscr.addch(y + 3, ornament_x + 2, "O", color | curses.A_BOLD)

        # Draw status info
        stdscr.addstr(sh - 2, 2, f"Speed: {scroll_speed:.2f}s  Status: {'PAUSED' if paused else 'RUNNING'}", curses.color_pair(3))
        stdscr.addstr(sh - 2, sw - 30, "Controls: [+] Faster [-] Slower [Space] Pause [q] Quit")

        stdscr.refresh()

        if not paused:
            scroll_pos = (scroll_pos + 1) % len(christmas_art[0])
            time.sleep(scroll_speed)
        else:
            time.sleep(0.1)

if __name__ == "__main__":