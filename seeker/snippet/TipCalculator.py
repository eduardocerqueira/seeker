#date: 2023-03-22T16:50:30Z
#url: https://api.github.com/gists/feb177769a417d9358d0f7e0a08dc1e5
#owner: https://api.github.com/users/t0023656

print("Welcome to the tip calculator.")
bill = float(input("What was the total bill? $"))
tipPercentage = float(input("What perentage tip would you like to give? 10, 12, or 15? "))
people = int(input("How many people to split the bill? "))

tip = bill + bill * tipPercentage / 100
pay = tip / people
print(f"Each person should pay: {round(pay, 2)}")