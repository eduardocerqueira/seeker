#date: 2024-06-03T17:07:40Z
#url: https://api.github.com/gists/9ebda0c061f31f7a28179e287bdc272a
#owner: https://api.github.com/users/wFirbolg

import tkinter as tk
from tkinter import ttk

# Define Orca Slicer theme colors
bg_color = "#3d4445"  # Black RGB(61,68,69)
button_color = "#419487"  # Cyan RGB(65,148,135)
text_color = "#ffffff"  # White for text

def calculate_new_flow_rate(old_rate, modifier):
    return old_rate * (100 + modifier) / 100

def calculate_pass1():
    try:
        old_rate = float(current_flow_rate_entry.get())
        pass1_result = float(pass1_result_var.get())

        new_rate_pass1 = calculate_new_flow_rate(old_rate, pass1_result)

        pass1_flow_rate_var.set(f"{new_rate_pass1:.3f}")
    except ValueError:
        pass1_flow_rate_var.set("Invalid input")

def calculate_pass2():
    try:
        new_rate_pass1 = float(pass1_flow_rate_var.get())
        pass2_result = float(pass2_result_var.get())

        new_rate_pass2 = calculate_new_flow_rate(new_rate_pass1, pass2_result)

        pass2_flow_rate_var.set(f"{new_rate_pass2:.3f}")
    except ValueError:
        pass2_flow_rate_var.set("Invalid input")

def set_pass1_modifier(modifier):
    pass1_result_var.set(modifier)
    calculate_pass1()

def set_pass2_modifier(modifier):
    pass2_result_var.set(modifier)
    calculate_pass2()

# Create the main window
root = tk.Tk()
root.title("Orca - Flow Rate Calibration")
root.configure(bg=bg_color)

# Create input fields
ttk.Label(root, text="Current Profile Flow Rate:", background=bg_color, foreground=text_color).grid(column=0, row=0, padx=10, pady=5)
current_flow_rate_entry = ttk.Entry(root)
current_flow_rate_entry.grid(column=1, row=0, padx=10, pady=5)

ttk.Label(root, text="Result of Pass 1 Calibration:", background=bg_color, foreground=text_color).grid(column=0, row=1, padx=10, pady=5)

pass1_result_var = tk.StringVar()
pass1_result_label = ttk.Label(root, textvariable=pass1_result_var, background=bg_color, foreground=text_color)
pass1_result_label.grid(column=1, row=1, padx=10, pady=5)

# Buttons for Pass 1 modifiers
pass1_buttons_frame = tk.Frame(root, bg=bg_color)
pass1_buttons_frame.grid(column=0, row=2, columnspan=2, pady=5)

pass1_modifiers = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
for mod in pass1_modifiers:
    button = tk.Button(pass1_buttons_frame, text=str(mod), command=lambda m=mod: set_pass1_modifier(m), bg=button_color, fg=text_color)
    button.pack(side=tk.LEFT, padx=2)

ttk.Label(root, text="New Flow Rate After Pass 1:", background=bg_color, foreground=text_color).grid(column=0, row=3, padx=10, pady=5)
pass1_flow_rate_var = tk.StringVar()
pass1_flow_rate_label = ttk.Label(root, textvariable=pass1_flow_rate_var, background=bg_color, foreground=text_color)
pass1_flow_rate_label.grid(column=1, row=3, padx=10, pady=5)

ttk.Label(root, text="Result of Pass 2 Calibration:", background=bg_color, foreground=text_color).grid(column=0, row=4, padx=10, pady=5)

pass2_result_var = tk.StringVar()
pass2_result_label = ttk.Label(root, textvariable=pass2_result_var, background=bg_color, foreground=text_color)
pass2_result_label.grid(column=1, row=4, padx=10, pady=5)

# Buttons for Pass 2 modifiers
pass2_buttons_frame = tk.Frame(root, bg=bg_color)
pass2_buttons_frame.grid(column=0, row=5, columnspan=2, pady=5)

pass2_modifiers = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
for mod in pass2_modifiers:
    button = tk.Button(pass2_buttons_frame, text=str(mod), command=lambda m=mod: set_pass2_modifier(m), bg=button_color, fg=text_color)
    button.pack(side=tk.LEFT, padx=2)

ttk.Label(root, text="New Flow Rate After Pass 2:", background=bg_color, foreground=text_color).grid(column=0, row=6, padx=10, pady=5)
pass2_flow_rate_var = tk.StringVar()
pass2_flow_rate_label = ttk.Label(root, textvariable=pass2_flow_rate_var, background=bg_color, foreground=text_color)
pass2_flow_rate_label.grid(column=1, row=6, padx=10, pady=5)

# Run the application
root.mainloop()
