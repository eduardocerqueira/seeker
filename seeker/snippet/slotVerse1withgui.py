#date: 2024-08-27T17:07:07Z
#url: https://api.github.com/gists/93e7e976f915d42bdfc710f9d38ae691
#owner: https://api.github.com/users/YigitChanson

import tkinter as tk
from tkinter import messagebox
import random

class CasinoAccount:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount

    def get_balance(self):
        return self.balance

account = CasinoAccount()

ROWS = 3
COLS = 3

symbol_count = {
    "A": 2,
    "B": 4,
    "C": 6,
    "D": 8
}

def get_slot_machine_spin(rows, cols, symbols):
    all_symbols = []
    for symbol, symbol_count in symbols.items():
        for i in range(symbol_count):
            all_symbols.append(symbol)

    slots = []
    for col in range(cols):
        slot = []
        current_symbols = all_symbols[:]
        for _ in range(rows):
            value = random.choice(current_symbols)
            current_symbols.remove(value)
            slot.append(value)

        slots.append(slot)

    return slots

def deposit():
    while True:
        amount = input("What would you like to deposit? (Maximum $300): ")
        if amount.isdigit():
            amount = int(amount)
            if 0 < amount <= 300:
                account.deposit(amount)
                print(f"Deposited ${amount}. Your current balance is ${account.get_balance()}.")
                break
            else:
                print("Invalid amount.")
        else:
            print("Invalid input. Please enter a number.")
    return account.get_balance()

def update_balance_label():
    balance_label.config(text=f"Balance: ${account.get_balance()}")

def place_bet():
    lines = int(lines_var.get())
    bet = int(bet_entry.get())
    total_bet = bet * lines

    if total_bet > account.get_balance():
        messagebox.showerror("Error", "You do not have enough balance to place this bet.")
        return

    account.balance -= total_bet
    update_balance_label()

    columns = get_slot_machine_spin(ROWS, COLS, symbol_count)
    for i in range(ROWS):
        for j in range(COLS):
            slot_labels[i][j].config(text=columns[j][i])

    # Slot sonuçlarını kontrol et (Bu kısım geliştirilebilir)
    messagebox.showinfo("Result", "Good luck next time!")

# Tkinter GUI
root = tk.Tk()
root.title("Casino Slot Machine")

balance_label = tk.Label(root, text=f"Balance: ${account.get_balance()}")
balance_label.pack()

deposit_button = tk.Button(root, text="Deposit $100", command=lambda: [account.deposit(100), update_balance_label()])
deposit_button.pack()

lines_var = tk.StringVar(value="1")
lines_label = tk.Label(root, text="Number of lines to bet on (1-3):")
lines_label.pack()
lines_entry = tk.Entry(root, textvariable=lines_var)
lines_entry.pack()

bet_label = tk.Label(root, text="Bet amount per line:")
bet_label.pack()
bet_entry = tk.Entry(root)
bet_entry.pack()

spin_button = tk.Button(root, text="Spin", command=place_bet)
spin_button.pack()

slot_labels = [[tk.Label(root, text="") for _ in range(COLS)] for _ in range(ROWS)]
for row in slot_labels:
    for label in row:
        label.pack()

root.mainloop()
