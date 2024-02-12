#date: 2024-02-12T17:09:18Z
#url: https://api.github.com/gists/d0b77554b1cde5f87b72a9a162c26cc3
#owner: https://api.github.com/users/sn0w2k

import curses
import psutil
import time

def monitor_system(stdscr, interval=1):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()

    while True:
        stdscr.refresh()

        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=interval)

        # Get memory usage
        mem = psutil.virtual_memory()
        mem_total = mem.total / (1024 * 1024)  # Convert to MB
        mem_used = mem.used / (1024 * 1024)    # Convert to MB
        mem_percent = mem.percent

        stdscr.addstr(0, 0, "System Monitor", curses.A_BOLD)
        stdscr.addstr(2, 0, f"CPU Usage: {cpu_percent:.2f}%", curses.color_pair(1))
        stdscr.addstr(3, 0, f"Memory Usage: {mem_used:.2f} MB / {mem_total:.2f} MB ({mem_percent}%)", curses.color_pair(2))

        stdscr.refresh()
        time.sleep(interval)

def main(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # CPU usage color
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Memory usage color

    monitor_system(stdscr)

if __name__ == "__main__":
    curses.wrapper(main)

