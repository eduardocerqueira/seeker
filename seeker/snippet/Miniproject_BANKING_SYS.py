#date: 2024-02-16T17:07:06Z
#url: https://api.github.com/gists/e31c99f1c5d0cecc7f1b1e9abe0d9e67
#owner: https://api.github.com/users/Saadsathr

#Banking system
account={'0980':'mizri@456','9876':'ashok@9876','6543':'gokul@6543'}

class Bank_account:


    def __init__(self,balance):
        self.balance=balance


    def deposit(self):
         amount=float(input("Enter the amount to be depsited: "))
         self.balance += amount
         print("\n Amount depsoited: ",amount)
    def withdraw(self):
        amount=float(input("Enter the amount to be withdrawn: "))
        if self.balance >= amount:
            self.balance -= amount
            print("\n You withdrew: ",amount)
        else:
            print("Insufficient Balance")
    def display(self):
        print(" \n Net available balance: ",self.balance)
def login():
    account_no = input("Enter your account_no: ")
    password = input("Enter your password: "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"c "**********"c "**********"o "**********"u "**********"n "**********"t "**********"_ "**********"n "**********"o "**********"  "**********"i "**********"n "**********"  "**********"a "**********"c "**********"c "**********"o "**********"u "**********"n "**********"t "**********"  "**********"a "**********"n "**********"d "**********"  "**********"a "**********"c "**********"c "**********"o "**********"u "**********"n "**********"t "**********"[ "**********"a "**********"c "**********"c "**********"o "**********"u "**********"n "**********"t "**********"_ "**********"n "**********"o "**********"] "**********"  "**********"= "**********"= "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
        print("Login succesfully..")
        balance=float(input("Enter your initial deposit: "))
        while True:
            print("\n choose 1 for deposit \n choose 2 for withdrawal \n choose 3 for account balance \n choose 4 for exit ")
            choice = int(input("choice: "))

            b = Bank_account(balance)
            if choice == 1:
                b.deposit()
            elif choice == 2:
                b.withdraw()
            elif choice == 3:
                b.display()
            elif choice==4:
                print("Have a nice day***")
                break
            else:
                print("Invalid error")


    else:
        print("Invalid username and password.Please try again.")
login()

