#date: 2021-11-11T17:12:33Z
#url: https://api.github.com/gists/0c0231526f8129ae558700238357d1a4
#owner: https://api.github.com/users/blackwer

import curses

def main(S):
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_BLACK)

    S.bkgdset(curses.color_pair(2))
    S.clear()
    S.addstr(0, 0, 'Hi', curses.color_pair(1))
    S.refresh()

    S.getch()

curses.wrapper(main)
