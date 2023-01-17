#date: 2023-01-17T16:47:16Z
#url: https://api.github.com/gists/7d5cd2fa99ff13c22a09f4efb735bcb1
#owner: https://api.github.com/users/SmeN3

import tkinter as tk
import os

WORD_TYPING_SPEED = 150

def get_input_file_path():
    """
    Get the path of the input text file
    """
    input_file_path = input_txt.get("1.0", "end-1c").strip()
    if os.path.isfile(input_file_path) and input_file_path.endswith('.txt'):
        with open(input_file_path, 'r') as file:
            words = file.read().split()
            output_txt.insert(tk.END, words)
    else:
        output_txt.insert(tk.END, "Error: Invalid file path or file format")
        return
    return words

def type_words(words):
    """
    Type the words of the text file one by one
    """
    if words:
        word = words.pop(0)
        word_label.config(text=word)
        root.after(WORD_TYPING_SPEED, type_words, words)
    else:
        word_label.config(text="All words typed")

# Create the main window
root = tk.Tk()
root.title("Text Typer")

# Create labels, text inputs and buttons
file_path_label = tk.Label(root, text="Set the directory for the text file")
input_txt = tk.Text(root, height=1, width=200, bg="light yellow")
output_txt = tk.Text(root, height=40, width=200, bg="light cyan")
display_btn = tk.Button(root, height=2, width=20, text="Show", command=get_input_file_path)
word_label = tk.Label(root, text="", height=10, width=10, font=('Helvetica bold', 26))
start_btn = tk.Button(root, height=2, width=20, text='Start', command=lambda: type_words(get_input_file_path()))

# Pack the widgets
file_path_label.pack()
input_txt.pack()
display_btn.pack()
output_txt.pack()
start_btn.pack()
word_label.pack()

root.mainloop()
