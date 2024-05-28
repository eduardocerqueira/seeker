#date: 2024-05-28T16:49:22Z
#url: https://api.github.com/gists/cfb74381e12ff6b24bb98b85fdac39ad
#owner: https://api.github.com/users/hirdle

import tkinter as tk
from tkinter import ttk
import random

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title('df')
        self.geometry('600x400')

        self.color_curr = tk.StringVar()
        self.color_curr.trace_add('write', self.bind_color)

        self.text_widget = tk.Text(height=5, width=40)
        self.text_widget.grid(row=0, column=2)

        btn_random = ttk.Button(text="Зарандомить", command=self.text_random)
        btn_random.grid(row=1, column=2)

        self.label = ttk.Label(text="please choose the color")
        self.label.grid(row=0, column=0)

        for idx, val in enumerate(['red', 'blue', 'orange', 'yellow', 'pink']):
            btn = ttk.Radiobutton(text=val, value=val, variable=self.color_curr)
            btn.grid(row=idx+1, column=0)

        btn = ttk.Button(text="Залить", command=self.fill_text)
        btn.grid(row=9, column=0)

    def bind_color(self, *args, **kwargs):
        self.label.grid_remove()

    def fill_text(self):
        self.text_widget['bg'] = self.color_curr.get()

    def text_random(self):
        self.text_widget.delete(1.0, tk.END)
        a = ord('а')
        self.text_widget.insert(1.0, ''.join(random.choices([chr(i) for i in range(a, a + 32)], k=40*5)))



app = App()
app.mainloop()