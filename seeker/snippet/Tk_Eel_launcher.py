#date: 2024-04-11T16:56:07Z
#url: https://api.github.com/gists/9e0300709e8f132cbe00b3451dd58323
#owner: https://api.github.com/users/YuSung-2022

from tkinter import *
from tkinter import font
import os, time

win = Tk()
win.minsize(200, 40)
win.maxsize(200, 40)
win.attributes('-topmost', 1)

font = font.Font(family="DisposableDroid BB", size=30)

def start_py():
	os.system("python app.py")
	return

btn = Button(win, text='START', command=start_py)
btn.config(width=100, font=font, relief=FLAT, activebackground='#f02468', activeforeground='white', bg='#3df2a4', fg='#323232')
btn.pack()

win.mainloop()