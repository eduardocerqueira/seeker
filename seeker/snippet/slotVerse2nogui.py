#date: 2024-08-27T17:08:41Z
#url: https://api.github.com/gists/5e1491aee45e618ca0422cd92fecda53
#owner: https://api.github.com/users/YigitChanson

import random 

class CasinoAccount:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount

    def get_balance(self):
        return self.balance

# Sınıfın dışına çıkarıldı
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

def print_slot_machine(slots):
    for row in range(len(slots[0])):
        for i, slot in enumerate(slots):
            if i != len(slots) - 1:
                print(slot[row], "|", end=" ")
            else:
                print(slot[row])

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

def get_number_of_lines():
    while True:
        lines = input("How many lines would you like to bet on? (1-3): ")
        if lines.isdigit() and 1 <= int(lines) <= 3:
            lines = int(lines)
            return lines
        else:
            print("Invalid input. Please enter a number between 1 and 3.")

def get_bet():
    while True:
        amount = input("What would you like to bet on each line? ($): ")
        if amount.isdigit():
            amount = int(amount)
            if amount >= 1:
                return amount
            else:
                print("The minimum bet is $1.")
        else:
            print("Invalid input. Please enter a number.")

def main():
    balance = deposit()
    lines = get_number_of_lines()
    bet = get_bet()
    total_bet = bet * lines
    if total_bet > balance:
        print(f"You do not have enough balance to place this bet. Your current balance is ${balance}.")
    else:
        print(f"You are betting ${bet} on {lines} lines. Total bet is ${total_bet}.")
        print(f"Your remaining balance after this bet is ${balance - total_bet}.")

        columns = get_slot_machine_spin(ROWS, COLS, symbol_count)
        print_slot_machine(columns)

main()