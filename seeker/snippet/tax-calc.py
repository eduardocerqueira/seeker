#date: 2024-09-25T16:54:06Z
#url: https://api.github.com/gists/870cf936f5197a8af0894d0bd6e2b05a
#owner: https://api.github.com/users/verbalius

# import time

# Tax brackets and rates based on the Portugal 2024 rates from PwC website
brackets = [
    # lower, upper, rate,   deductible
    (0,      7703,  0.1325, 0),
    (7703,   11623, 0.18,   365.89),
    (11623,  16472, 0.23,   947.04),
    (16472,  21321, 0.26,   1441.14),
    (21321,  27146, 0.3275, 2880.47),
    (27146,  39791, 0.37,   4034.17),
    (39791,  51997, 0.435,  6620.43),
    (51997,  81199, 0.45,   7400.21)
]

print("🤓 This calculator will help you determine your net income after taxes in Portugal as a recibo verde self-employed worker.")
# time.sleep(4)

# read income from the user
income = int(input("🤫 Enter your yearly expected income (format 12345): "))
assert income > 0, "🤬 Income must be a positive number"
deductions = int(input("🤫 Enter your yearly expected deductions [rent, food, etc] (format 12345): ") or 0)
assert deductions >= 0, "🤬 Deductions must be a positive number"
if income < 6122.64:
    print("You are below the minimum wage in Portugal. You do not need to pay taxes! 🤑")
    exit()
elif income > 50000:
    print("Good job! You are quite above the minimum wage in Portugal. You need to pay taxes, yaay 🥳 You'd better sit down! 🚑")

total_tax = 0

nhr = input("🤫 Do you have NHR rate? (yes/no): ") or "no"
if nhr in ["yes", "нуі"]:
    print("🤑 You have NHR rate. You only pay 20% flat tax")
    total_tax = round(income * 0.2, 2)
else:
    print("😿 You do not have NHR rate. You will pay progressive tax")
    print("🧮 Calculating total progressive IRS tax.")
    for lower, upper, rate, deductible in brackets:
        if income > lower:
            taxable = min(income, upper) - lower
            tax = round(taxable * rate - min(deductible, deductions), 2)
            total_tax += tax
            deductions -= min(deductible, deductions)
            print("[Taxes payed:", round(total_tax, 2), "]", "For bracket", lower, "-", upper, "taxable income is", taxable, "tax rate is", rate*100, "%", "tax paid", tax)

# Net income after taxes
net_income = round(income - total_tax, 2)
print("Yayy! 💸💸💸 IRS taxes paid:", round(total_tax, 2), "EUR which is", round(total_tax / income * 100, 2), "% of your income")

# Social Security contributions
print("🏥 Now let's calculate your social security contributions.")
print("🧮 Calculating social security contributions.")
first_year = input("🤫 Is this your first year as a self-employed worker in Portugal? (yes/no): ") or "no"
if first_year in ["yes", "нуі"]:
    social_security_tax_rate = 0
else:
    social_security_tax_rate = 0.214

print("😀 Cool, your social security tax rate is", social_security_tax_rate * 100, "%")

# The value of the provision of services is considered at 70% of the relevant income.
socially_taxable_income = income * 0.7
print("Good news! You only need pay social tax from 70% of your income.")
social_security_tax = round(income * 0.7 * social_security_tax_rate, 2)
print("Yayy! 💸💸💸 Social security contributions paid:", social_security_tax, "EUR which is", round(social_security_tax / income * 100, 2), "% of your income")
net_income = net_income - social_security_tax

print("Drum roll please 🥁🥁🥁")

print("🎉🎉🎉 After all the taxes and social security contributions you are left with", round(net_income,2), "EUR 🎉🎉🎉")

print("Portugal is a beautiful country, and it costs you", round(income - net_income, 2), "EUR a year which is", round((1 - (net_income / income)) * 100, 2), "% of your income")

print("Beijinhos, tchau tchau, tchau! 😘")
