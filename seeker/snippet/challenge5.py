#date: 2023-08-03T16:55:11Z
#url: https://api.github.com/gists/fb20f376a676a4776efbabe67e42258e
#owner: https://api.github.com/users/greatvijay

#Challenge 5
class Account:
    def __init__(self, title=None, balance=0):
        self.title = title
        self.balance = balance

    def withdrawal(self, amount):
        self.balance -= amount

    def deposit(self, amount):
        self.balance += amount

    def getBalance(self):
        return self.balance

class SavingsAccount(Account):
    def __init__(self, title=None, balance=0, interestRate=0):
        super().__init__(title, balance)
        self.interestRate = interestRate

    def interestAmount(self):
        return (self.balance * self.interestRate) / 100

#code to test - do not edit this

demo1 = SavingsAccount("Ashish", 2000, 5)   # initializing a SavingsAccount object

# Testing the methods
demo1.deposit(500)
print(demo1.getBalance())     

demo1.withdrawal(500)
print(demo1.getBalance())      

print(demo1.interestAmount())  

