#date: 2024-04-03T16:51:22Z
#url: https://api.github.com/gists/c39855240bbe43fc048ea3912aa4c20d
#owner: https://api.github.com/users/SalsaMarz

import tkinter as tk
from tkinter import messagebox

def save_customer_details():
    name = entry_name.get()
    account_number = entry_account_number.get()
    balance = entry_balance.get()
    address = entry_address.get()

    if name.strip() == '' or account_number.strip() == '' or balance.strip() == '' or address.strip() == '':
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    try:
        with open("customer_details.txt", "a") as file:
            file.write(f"Name: {name}, Account Number: {account_number}, Balance: {balance}, Address: {address}\n")
        messagebox.showinfo("Success", "Customer details saved successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def read_customer_details():
    try:
        with open("customer_details.txt", "r") as file:
            details = file.read()
        if details.strip() == '':
            messagebox.showinfo("Info", "No customer details found.")
        else:
            messagebox.showinfo("Customer Details", details)
    except FileNotFoundError:
        messagebox.showinfo("Info", "No customer details found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# GUI setup
root = tk.Tk()
root.title("Bank Customer Details")

# Labels
label_name = tk.Label(root, text="Name:")
label_name.grid(row=0, column=0, padx=10, pady=5)
label_account_number = tk.Label(root, text="Account Number:")
label_account_number.grid(row=1, column=0, padx=10, pady=5)
label_balance = tk.Label(root, text="Balance:")
label_balance.grid(row=2, column=0, padx=10, pady=5)
label_address = tk.Label(root, text="Address:")
label_address.grid(row=3, column=0, padx=10, pady=5)

# Entry fields
entry_name = tk.Entry(root)
entry_name.grid(row=0, column=1, padx=10, pady=5)
entry_account_number = tk.Entry(root)
entry_account_number.grid(row=1, column=1, padx=10, pady=5)
entry_balance = tk.Entry(root)
entry_balance.grid(row=2, column=1, padx=10, pady=5)
entry_address = tk.Entry(root)
entry_address.grid(row=3, column=1, padx=10, pady=5)

# Buttons
button_save = tk.Button(root, text="Save", command=save_customer_details)
button_save.grid(row=4, column=0, padx=10, pady=5)
button_read = tk.Button(root, text="Read Details", command=read_customer_details)
button_read.grid(row=4, column=1, padx=10, pady=5)

root.mainloop()
