#date: 2024-05-20T17:11:13Z
#url: https://api.github.com/gists/0df552e2130cb419390827e35743950b
#owner: https://api.github.com/users/wgordy-infini

import tkinter as tk
from tkinter import messagebox

class CountdownTimer:
    def __init__(self, root):
        self.root = root
        self.root.title("Countdown Timer")

        self.time_left = 0
        self.running = False
        self.counter = 0

        self.timer_label = tk.Label(root, text="00:00", font=("Helvetica", 48))
        self.timer_label.pack()

        self.counter_label = tk.Label(root, text="Counter: 0", font=("Helvetica", 24))
        self.counter_label.pack()

        self.start_entry = tk.Entry(root, width=20)
        self.start_entry.insert(0, "12:00")
        self.start_entry.pack()

        self.reset_entry = tk.Entry(root, width=20)
        self.reset_entry.insert(0, "12:00")
        self.reset_entry.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start)
        self.start_button.pack(side="left")

        self.stop_button = tk.Button(root, text="Stop", command=self.stop)
        self.stop_button.pack(side="left")

        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack(side="left")

        self.update_timer()

    def start(self):
        if not self.running:
            try:
                self.time_left = self.parse_time(self.start_entry.get())
                self.reset_time = self.parse_time(self.reset_entry.get())
                self.original_time = self.time_left
                self.running = True
                self.update_timer()
            except ValueError:
                messagebox.showerror("Invalid input", "Please enter a valid time in mm:ss format")

    def stop(self):
        self.running = False

    def reset(self):
        self.running = False
        self.time_left = self.reset_time
        self.counter = 0
        self.update_display()

    def parse_time(self, time_str):
        mins, secs = map(int, time_str.split(':'))
        return mins * 60 + secs

    def format_time(self, total_seconds):
        mins, secs = divmod(total_seconds, 60)
        return f"{mins:02}:{secs:02}"

    def update_timer(self):
        if self.running and self.time_left > 0:
            self.time_left -= 1
            self.update_display()
            self.root.after(1000, self.update_timer)
        elif self.time_left == 0 and self.running:
            self.counter += 1
            self.counter_label.config(text=f"Counter: {self.counter}")
            self.time_left = self.reset_time
            self.update_display()
            self.root.after(1000, self.update_timer)

    def update_display(self):
        self.timer_label.config(text=self.format_time(self.time_left))

if __name__ == "__main__":
    root = tk.Tk()
    app = CountdownTimer(root)
    root.mainloop()