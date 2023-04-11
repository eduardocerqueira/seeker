#date: 2023-04-11T17:02:25Z
#url: https://api.github.com/gists/9e011e3ad0596fc560df45b77cc76c3c
#owner: https://api.github.com/users/echomalik

import tkinter as tk

window = tk.Tk()
window.title("Odd or Even")

def checkNumber():
	number = int(ent_number.get())

	if number % 2 == 0:
		lbl_result['text'] = f"{number} is an even number"
		ent_number.delete(0, tk.END)
	else:
		lbl_result['text'] = f"{number} is an odd number"
		ent_number.delete(0, tk.END)

lbl_title = tk.Label(text = "Odd or Even Number Checker")
ent_number = tk.Entry(font = ("Arial", 16))
btn_check = tk.Button(text = "Check", command = checkNumber)
lbl_result = tk.Label(font = ("Arial", 16))

lbl_title.pack()
ent_number.pack()
btn_check.pack()
lbl_result.pack()

window.mainloop()