#date: 2023-08-03T16:55:11Z
#url: https://api.github.com/gists/fb20f376a676a4776efbabe67e42258e
#owner: https://api.github.com/users/greatvijay

#Challenge 4
class Account:
    def __init__(self, title, balance):
        self.title = title
        self.balance = balance

class SavingsAccount(Account):
    def __init__(self, title, balance, interestRate):
        super().__init__(title, balance)
        self.interestRate = interestRate

# Testing the Account and SavingsAccount classes
account1 = Account("Ashish", 5000)
print(account1.title)  # Output: Ashish
print(account1.balance)  # Output: 5000

savings_account1 = SavingsAccount("Ashish", 5000, 5)
print(savings_account1.title)  
print(savings_account1.balance)  
print(savings_account1.interestRate)  
