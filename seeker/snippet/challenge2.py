#date: 2023-08-03T16:55:11Z
#url: https://api.github.com/gists/fb20f376a676a4776efbabe67e42258e
#owner: https://api.github.com/users/greatvijay

#Challenge 2
class Calculator:

    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2

    def add(self):
        return self.num1 + self.num2

    def subtract(self):
        return self.num2 - self.num1

    def multiply(self):
        return self.num1 * self.num2

    def divide(self):
        return self.num2 / self.num1

# Testing the Calculator class
obj = Calculator(10, 94)
print(obj.add())      
print(obj.subtract()) 
print(obj.multiply())  
print(obj.divide())   