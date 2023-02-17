#date: 2023-02-17T16:55:41Z
#url: https://api.github.com/gists/136c0e820537ea6f0860bc0144fd9261
#owner: https://api.github.com/users/prasadph

import glob
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk


class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title('Tkinter PhotoImage Demo')
        self.time = 20
        self.base_width = 1100
        # create fields
        self.prev_button = tk.Button(self, text="Prev", font=("fixed", 20, "bold"), command=self.prev_pic)
        self.next_button = tk.Button(self, text="Next", font=("fixed", 20, "bold"), command=self.next_pic)
        self.timer = ttk.Label(self, text=self.time, font=("fixed", 80, "bold"))
        self.image_view = ttk.Label(self)

        self.image_list = [Path(img) for img in glob.glob('./images/*.png')]
        self.current_image_id = 0
        self.max_image_id = len(self.image_list) - 1

        self.image_view.pack()
        self.change_pic(self.image_list[self.current_image_id])
        self.prev_button.pack(side="left")
        self.next_button.pack(side="left")
        self.timer.pack(side="right")

        self.after(1000, self.update)  # start the update 1 second later

    def change_pic(self, image_path):
        self.image = Image.open(Path(image_path))
        w, h = self.image.size
        new_height = int(self.base_width / w * h)
        self.image = self.image.resize((self.base_width, new_height))
        self.python_image = ImageTk.PhotoImage(self.image)
        self.image_view.configure(image=self.python_image)
        self.time = 20
        self.timer.configure(text=self.time)

    def next_pic(self):
        if self.current_image_id == self.max_image_id:
            return
        self.current_image_id += 1
        self.change_pic(self.image_list[self.current_image_id])

    def prev_pic(self):
        if self.current_image_id == 0:
            return
        self.current_image_id -= 1
        self.change_pic(self.image_list[self.current_image_id])

    def update(self):
        self.time -= 1
        self.timer.configure(text=self.time)
        self.after(1000, self.update)


if __name__ == '__main__':
    app = App()
    app.mainloop()
