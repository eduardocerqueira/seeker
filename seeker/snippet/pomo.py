#date: 2024-08-27T16:59:34Z
#url: https://api.github.com/gists/fbf95eab2ac2c6fa692603254e28aa45
#owner: https://api.github.com/users/YigitChanson

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # tkinter'in standart ttk modülü
import time

WORK_TIME = 25 * 60
SHORT_BREAK_TIME = 5 * 60
LONG_BREAK_TIME = 15 * 60

class PomodoroTimer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("300x300")
        self.root.title("Pomodoro Timer")

        self.timer_label = tk.Label(self.root, text="00:00", font=("TkDefaultFont", 40))
        self.timer_label.pack(pady=20)

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_timer)
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_timer, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.reset_button = ttk.Button(self.root, text="Reset", command=self.reset_timer, state=tk.DISABLED)
        self.reset_button.pack(pady=5)

        self.work_time, self.break_time = WORK_TIME, SHORT_BREAK_TIME
        self.is_work_time, self.pomodoros_completed, self.is_running = True, 0, False

        self.root.mainloop()

    def start_timer(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.is_running = True
        self.update_timer()

    def stop_timer(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL)
        self.is_running = False

    def reset_timer(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.work_time, self.break_time = WORK_TIME, SHORT_BREAK_TIME
        self.is_work_time = True
        self.timer_label.config(text="00:00")

    def update_timer(self):
        if self.is_running:
            if self.is_work_time:
                self.work_time -= 1
                minutes, seconds = divmod(self.work_time, 60)
                self.timer_label.config(text=f"{minutes:02}:{seconds:02}")
                if self.work_time == 0:
                    self.is_work_time = False
                    self.pomodoros_completed += 1
                    self.break_time = LONG_BREAK_TIME if self.pomodoros_completed % 4 == 0 else SHORT_BREAK_TIME
                    messagebox.showinfo(
                        "Good job!",
                        "Take a long break to rest your mind." if self.pomodoros_completed % 4 == 0 else "Take a short break and stretch your legs"
                    )
            else:
                self.break_time -= 1
                minutes, seconds = divmod(self.break_time, 60)
                self.timer_label.config(text=f"{minutes:02}:{seconds:02}")
                if self.break_time == 0:
                    self.is_work_time = True
                    self.work_time = WORK_TIME
                    messagebox.showinfo("Break over!", "Time to get back to work!")

            self.root.after(1000, self.update_timer)


PomodoroTimer()