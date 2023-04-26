#date: 2023-04-26T16:50:19Z
#url: https://api.github.com/gists/7fe1f9a6d6b444cb53121a9ce04fd971
#owner: https://api.github.com/users/Zekeriyaexe

from tkinter import Label, Tk
import time
window = Tk()
window.title("Python Dijital Saat - diveebi.com")
window.geometry("490x180")
window.resizable(1,1)

text= ("Roboto", 80, 'bold')
background = "#15bf9e"
foreground= "#f1f1f1"
border = 30

label = Label(window, font=text, bg=background, fg=foreground, bd=border)
label.grid(row=0, column=1)

def digitalClock():
   time_live = time.strftime("%H:%M:%S")
   label.config(text=time_live)
   label.after(200, digitalClock)

digitalClock()
window.mainloop()