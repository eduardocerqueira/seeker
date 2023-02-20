#date: 2023-02-20T16:49:59Z
#url: https://api.github.com/gists/bf32ce34d8b7da458f7d29d522ffdfe6
#owner: https://api.github.com/users/ysnefndyctn

print("Welcome to the tip calculaltor!")
bill=float(input("What was the total bill?: "))
tip=float(input("how much tip would you like to give ?10,12 or 15 \n"))
total_bill= tip/100 * bill+bill
ppl=int(input("how many people will split the bill: "))
bill_per_person=total_bill/ppl
final_amount= round(bill_per_person,2)
final_amount="{:.2f}".format(bill_per_person)
print(f"Each person should pay: {final_amount} dollars")