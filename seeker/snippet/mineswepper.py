#date: 2023-02-02T17:07:33Z
#url: https://api.github.com/gists/a3748d4b6624abf644ede41954b99251
#owner: https://api.github.com/users/edoigtrd

#!/usr/bin/env python3
from curses import *
import random
import sys

class config :
    xsize = 18
    ysize = 18
    mines_count = 40
    colors = [COLOR_CYAN,COLOR_BLUE, COLOR_GREEN, COLOR_MAGENTA, COLOR_RED, COLOR_YELLOW, COLOR_WHITE]
    flag_color = COLOR_RED

# help message
if len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help","help"] :
    print("Usage: python3 main.py [xsize] [ysize] [mines_count]")
    print("Default: xsize = 18, ysize = 18, mines_count = 40")
    sys.exit()


if len(sys.argv) == 4 :
    config.xsize = int(sys.argv[1])
    config.ysize = int(sys.argv[2])
    config.mines_count = int(sys.argv[3])

class cursor :
    x = int(config.xsize / 2)
    y = int(config.ysize / 2)

gameover = False
board = [[0 for x in range(config.xsize)] for y in range(config.ysize)]
view_board = [[0 for x in range(config.xsize)] for y in range(config.ysize)]
neighbour_board = [[0 for x in range(config.xsize)] for y in range(config.ysize)]

def init_board() :
    for i in range(config.mines_count) :
        x = random.randint(0, config.xsize - 1)
        y = random.randint(0, config.ysize - 1)
        board[x][y] = -1

def get_neighbor_number(x,y) :
    count = 0
    for i in range(-1, 2) :
        for j in range(-1, 2) :
            if x + i >= 0 and x + i < config.xsize and y + j >= 0 and y + j < config.ysize :
                if board[x + i][y + j] == -1 :
                    count += 1
    return count

def init_neighbour_board() :
    for x in range(config.xsize) :
        for y in range(config.ysize) :
            if board[x][y] != -1 :
                neighbour_board[x][y] = get_neighbor_number(x,y)
            else :
                neighbour_board[x][y] = -1

def init_color_pairs() :
    start_color()
    for i in range(len(config.colors)) :
        init_pair(i + 1, config.colors[i], COLOR_BLACK)
    init_pair(len(config.colors) + 1, config.flag_color, COLOR_BLACK)
    

init_board()
init_neighbour_board()


def update_view_board(x,y) :
    if view_board[x][y] == 0 :
        if board[x][y] == -1 :
            view_board[x][y] = 1
            return False
        elif neighbour_board[x][y] > 0 :
            view_board[x][y] = 1
        else :
            view_board[x][y] = 1
            for i in range(-1, 2) :
                for j in range(-1, 2) :
                    if x + i >= 0 and x + i < config.xsize and y + j >= 0 and y + j < config.ysize :
                        update_view_board(x + i, y + j)
    return True

def draw_board(stdscr, show_mines = False) :
    for y in range(config.ysize) :
        for x in range(config.xsize) :
            if show_mines and board[x][y] == -1 :
                stdscr.addstr(y, x, "X", color_pair(len(config.colors) + 1))
            if view_board[x][y] == 0:
                stdscr.addstr(y, x, "□")
            elif view_board[x][y] == 1 :
                if neighbour_board[x][y] > 0 :
                    stdscr.addstr(y, x, str(neighbour_board[x][y]), color_pair(neighbour_board[x][y]))
                else :
                    stdscr.addstr(y, x, " ")
            elif view_board[x][y] == 2 :
                stdscr.addstr(y, x, "F", color_pair(len(config.colors) + 1))

def check_win() :
    if str(view_board).count("2") != config.mines_count :
        return False
    for x in range(config.xsize) :
        for y in range(config.ysize) :
            if view_board[x][y] == 2 and board[x][y] != -1 :
                return False
    return True
    


def main(stdscr) :
    global gameover
    init_board()
    init_color_pairs()
    stdscr.nodelay(True)
    stdscr.keypad(True)
    stdscr.clear()
    draw_board(stdscr)
    while True :
        if check_win() :
            stdscr.addstr(0, 0, "You Win!")
            stdscr.getch()
            continue
        stdscr.move(cursor.y, cursor.x)
        key = stdscr.getch()
        if key == ord("q") :
            stdscr.clear()
            break
        elif key == ord("f") :
            stdscr.clear()
            if view_board[cursor.x][cursor.y] == 0 :
                view_board[cursor.x][cursor.y] = 2
            elif view_board[cursor.x][cursor.y] == 2 :
                view_board[cursor.x][cursor.y] = 0
            
        elif key == KEY_UP :
            stdscr.clear()
            if cursor.y > 0 :
                cursor.y -= 1
        elif key == KEY_DOWN :
            stdscr.clear()
            if cursor.y < config.ysize - 1 :
                cursor.y += 1
        elif key == KEY_LEFT :
            stdscr.clear()
            if cursor.x > 0 :
                cursor.x -= 1
        elif key == KEY_RIGHT :
            stdscr.clear()
            if cursor.x < config.xsize - 1 :
                cursor.x += 1
        elif key == ord(" ") :
            stdscr.clear()
            if update_view_board(cursor.x, cursor.y) == False :
                gameover = True
                break
        draw_board(stdscr, gameover)
    if gameover :
        stdscr.addstr(0, 0, "You Lose!")
        draw_board(stdscr, True)
    stdscr.getch()

if __name__ == "__main__" :
    wrapper(main)
    print("Game Over")