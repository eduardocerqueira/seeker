#date: 2023-12-25T16:41:36Z
#url: https://api.github.com/gists/4a2e4db717670669aadabf86e6190854
#owner: https://api.github.com/users/secemp9

import tkinter as tk
from tkinter import ttk

tab_count = 1

def create_tab():
    global tab_count  # Access the global tab_count variable
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=f"<Untitled> {tab_count}")  # Use tab_count in the tab text
    text_widget = tk.Text(tab)
    text_widget.pack(fill="both", expand=True)
    notebook.select(tab)
    text_widget.focus_set()
    tab_count += 1  # Increment the tab_count

def switch_tab(event=None):
    current_tab = notebook.select()
    notebook.select(current_tab)
    current_tab_widget = notebook.nametowidget(current_tab)
    text_widget = current_tab_widget.winfo_children()[0]
    text_widget.focus_set()

root = tk.Tk()
root.title("Text Editor")

frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)

notebook = ttk.Notebook(frame)
notebook.pack(fill="both", expand=True)

create_tab()  # Create the initial tab

add_button = ttk.Button(frame, text="New Tab", command=create_tab)
add_button.pack(side="bottom")

notebook.bind("<ButtonRelease-1>", switch_tab)

root.mainloop()
