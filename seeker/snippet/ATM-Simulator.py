#date: 2026-03-02T17:36:21Z
#url: https://api.github.com/gists/b1663738ce7638249b6017e8ee976be2
#owner: https://api.github.com/users/hannaDesalegn

#### ADVANCED ATM SIMULATOR ####

"""
FEATURES
1. Multiple users with unique account numbers and PINs
2. Pin verifications (3 trials before account lock)
3. Balance inquiry
4. Cash withdrawal with balance checks
5. Cash deposit
6. Fund transfer 
7. Transaction history
8. Change pin
9. Exit Safely
"""
print("HELLO TEST")
# ================================
# STORE USERS IN A LIST OF DICTIONARIES
# ================================

users = [
    {"username": "John", "full_name": "John Doe", "account_number": "1234567890", "pin": "1234", "balance": 10000.0, "transaction_history": []},
    {"username": "Ayomide", "full_name": "Ayomide Smith", "account_number": "0987654321", "pin": "5678", "balance": 15000.0, "transaction_history": []},
    {"username": "Samuel", "full_name": "Samuel Johnson", "account_number": "1122334455", "pin": "4321", "balance": 20000.0, "transaction_history": []},
    {"username": "Bob", "full_name": "Bob Brown", "account_number": "5566778899", "pin": "8765", "balance": 5000.0, "transaction_history": []},
    {"username": "Charlie", "full_name": "Charlie Davis", "account_number": "6677889900", "pin": "2468", "balance": 8000.0, "transaction_history": []}
]

# ================================
# FUNCTIONS
# ================================

def find_user(username):
    for user in users:
        if user["username"].lower() == username.lower():
            return user
    return None


def login():
    entered_username = input("Enter your username: ")
    user_info = find_user(entered_username)

    if user_info is None:
        print("User not found!")
        return login()

    trials = 3
    while trials > 0:
        pin = input("Enter your pin: ")
        if pin == user_info["pin"]:
            print("Login Successful!\n")
            return user_info
        else:
            trials -= 1
            print("Incorrect Pin. Attempts left:", trials)

    print("Too many failed attempts. Account locked.")
    return login()


def check_balance(user_info):
    print(f"Current Balance: N{user_info['balance']}")


def deposit(user_info):
    amount = input("Enter amount to deposit: ")
    if amount.isdigit():
        amount = int(amount)
        if amount > 0:
            user_info["balance"] += amount
            user_info["transaction_history"].append(f"Deposited N{amount}")
            print("Deposit successful!")
        else:
            print("Amount must be greater than 0")
    else:
        print("Invalid amount")


# ================================
# TASK 1: withdraw()
# ================================

def withdraw(user_info):
    amount = input("Enter amount to withdraw: ")

    if amount.isdigit():
        amount = int(amount)

        if amount <= 0:
            print("Amount must be greater than 0")

        elif amount > user_info["balance"]:
            print("Insufficient balance")

        else:
            user_info["balance"] -= amount
            user_info["transaction_history"].append(f"Withdrew N{amount}")
            print(f"N{amount} withdrawn successfully")

    else:
        print("Invalid amount")


# ================================
# TASK 2: transfer()
# ================================

def transfer(user_info):
    receiver_name = input("Enter receiver's username: ")
    receiver = find_user(receiver_name)

    if receiver is None:
        print("Receiver not found")
        return

    amount = input("Enter amount to transfer: ")

    if amount.isdigit():
        amount = int(amount)

        if amount <= 0:
            print("Amount must be greater than 0")

        elif amount > user_info["balance"]:
            print("Insufficient balance")

        else:
            user_info["balance"] -= amount
            receiver["balance"] += amount

            user_info["transaction_history"].append(
                f"Transferred N{amount} to {receiver['username']}"
            )

            receiver["transaction_history"].append(
                f"Received N{amount} from {user_info['username']}"
            )

            print("Transfer successful")

    else:
        print("Invalid amount")


# ================================
# TASK 3: change_pin()
# ================================

def change_pin(user_info):
    new_pin = input("Enter new 4-digit PIN: ")

    if len(new_pin) == 4 and new_pin.isdigit():
        user_info["pin"] = new_pin
        print("PIN updated successfully")
    else:
        print("Invalid PIN. Must be exactly 4 digits")


# ================================
# TASK 4: view_transaction_history()
# ================================

def view_transaction_history(user_info):
    if not user_info["transaction_history"]:
        print("No transactions yet")
    else:
        print("Transaction History:")
        for transaction in user_info["transaction_history"]:
            print("-", transaction)


# ================================
# MAIN PROGRAM
# ================================

def main():
    print("======== Welcome to Optimum Bank ATM ========")

    current_user = login()

    if current_user:
        while True:
            print("\n====== MAIN MENU ======")
            print("1. Check Balance")
            print("2. Deposit")
            print("3. Withdraw")
            print("4. Transfer")
            print("5. Change Pin")
            print("6. View Transaction History")
            print("7. Exit")

            choice = input("Select option (1-7): ")

            if choice == "1":
                check_balance(current_user)

            elif choice == "2":
                deposit(current_user)

            elif choice == "3":
                withdraw(current_user)

            elif choice == "4":
                transfer(current_user)

            elif choice == "5":
                change_pin(current_user)

            elif choice == "6":
                view_transaction_history(current_user)

            elif choice == "7":
                print("Thank you for using Optimum Bank ATM.")
                break

            else:
                print("Invalid option!")


main()