#date: 2023-04-17T16:54:58Z
#url: https://api.github.com/gists/cb477926629074fc75758c52a7c9c1c3
#owner: https://api.github.com/users/t0023656

import math
from tkinter import *

PINK = "#e2979c"
RED = "#e7305b"
GREEN = "#9bdeac"
YELLOW = "#f7f5dd"
FONT_NAME = "Courier"
WORK_MIN = 25
SHORT_BREAK_MIN = 5
LONG_BREAK_MIN = 15
CANVAS_WIDTH = 210
CANVAS_HEIGHT = 224

cycles = 0
is_working = False
is_count_down_finish = True
timer = None


########### Function ########
def count_down(current_time):
    global timer

    show_time(current_time)
    if current_time > 0:
        timer = window.after(1000, count_down, current_time - 1)
    else:
        finish_count_down()


def show_time(current_time):
    timer_min = math.floor(current_time / 60)
    timer_sec = current_time % 60

    # 設定秒數格式為前面有0（例：09），而不是只有 9 一個數字
    if timer_sec < 10:
        show_timer_sec = f"0{timer_sec}"
    else:
        show_timer_sec = timer_sec

    canvas.itemconfig(timer_text, text=f"{timer_min}:{show_timer_sec}")


def show_cycle():
    message = ""
    for i in range(cycles):
        message = f"{message} ✔ "
    cycles_label.config(text=message)


def start_timer():
    global is_working
    global is_count_down_finish
    global cycles

    # 如果還在倒數，按下 start 無作用
    if is_count_down_finish:
        is_count_down_finish = False
        if not is_working:
            if cycles >= 4:
                cycles = 0
                show_cycle()
            start_work()
        else:
            start_rest()


def start_work():
    global is_working

    is_working = True
    timer_label.config(text="Work", fg=GREEN)
    count_down(WORK_MIN * 60)


def start_rest():
    global is_working

    is_working = False
    if cycles % 4 == 0:
        timer_label.config(text="Break", fg=RED)
        count_down(LONG_BREAK_MIN * 60)
    else:
        timer_label.config(text="Break", fg=PINK)
        count_down(SHORT_BREAK_MIN * 60)


def finish_count_down():
    global cycles
    global is_count_down_finish

    is_count_down_finish = True
    if is_working:
        cycles += 1
        if cycles % 4 == 0:
            show_time(LONG_BREAK_MIN * 60)
        else:
            show_time(SHORT_BREAK_MIN * 60)
    else:
        show_time(WORK_MIN * 60)
    show_cycle()


def reset_timer():
    global is_working
    global cycles

    if timer:
        window.after_cancel(timer)
    is_working = False
    cycles = 0
    timer_label.config(text="Timer", fg=GREEN)
    finish_count_down()


########### set UI ##########
# set window
window = Tk()
window.title("Pomodoro")
window.config(padx=100, pady=50, bg=YELLOW)

# set canvas and images
canvas = Canvas(width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg=YELLOW, highlightthickness=0)
tomato_img = PhotoImage(file="tomato.png")
canvas.create_image(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, image=tomato_img)
timer_text = canvas.create_text(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 20, text="25:00", fill="white",
                                font=(FONT_NAME, 35, "bold"))
canvas.grid(row=1, column=1)

# set labels
timer_label = Label(text="Timer", font=(FONT_NAME, 50))
timer_label.config(bg=YELLOW, fg=GREEN)
timer_label.grid(row=0, column=1)

cycles_label = Label(text="", bg=YELLOW, fg=GREEN, font=(FONT_NAME, 18))
cycles_label.grid(row=3, column=1)

# set buttons
start_button = Button(text="Start", bg=YELLOW, highlightthickness=0, command=start_timer)
start_button.grid(row=2, column=0)

reset_button = Button(text="Reset", bg=YELLOW, highlightthickness=0, command=reset_timer)
reset_button.grid(row=2, column=2)

window.mainloop()
