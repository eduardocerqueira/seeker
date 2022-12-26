#date: 2022-12-26T16:23:14Z
#url: https://api.github.com/gists/97b87b80da7f86290d6068dddd3b5d07
#owner: https://api.github.com/users/asmittiwari50-cmd

def initial_display():
    print('''
                  Sunway College Account Department
                          Maitidevi,Kathmandu
                              Welcome to
                  Salary Tax Calculation System (STCS) 
            ----------------------------------------------------
    ''')


def calculate_tax_of_staff(salary):
    income = salary * 12
    if (income <= 400000):
        tax_amount = income * 0.01
    elif (income > 400000 and income <= 500000):
        tax_amount = (400000 * 0.01) + ((income - 400000) * 0.10)
    elif (income > 400000 and income <= 700000):
        tax_amount = (400000 * 0.01) + ((100000) * 0.10) + (income - 500000) * 0.20
    elif (income > 400000 and income <= 1300000):
        tax_amount = (400000 * 0.01) + ((100000) * 0.10) + ((200000) * 0.20) + (income - 700000) * 0.30
    elif (income >= 2000000):
        tax_amount = (400000 * 0.01) + (income - 400000 * 0.36)
    return tax_amount


def calculate_tax_of_married(salary):
    income = salary * 12
    if (income <= 450000):
        tax_amount = income * 0.01
    elif (income > 450000 and income <= 550000):
        tax_amount = (450000 * 0.01) + ((income - 450000) * 0.10)
    elif (income > 450000 and income < 750000):
        tax_amount = (450000 * 0.01) + (100000 * 0.10) + (income - 550000) * 0.20
    elif (income > 450000 and income < 1300000):
        tax_amount = (450000 * 0.01) + (100000 * 0.10) + (200000 * 0.20) + (income - 7500000) * 0.30
    elif (income > 2000000):
        tax_amount = (400000 * 0.01) + (income - 400000 * 0.36)
    return tax_amount


def display_staff_info(name, address, pan, salary, taxamount, foryear, married):
    initial_display()
    print(f'''
     Name of the staff: {name}\t\t\t\tAddress: {address}
     PAN no: {pan}\t\t\t\t\tFor Year:{foryear}\t\t\tMarried Status: {married}
    ''')
    if (salary * 12 <= 400000):
        slab = "1%"
    elif (salary * 12 <= 500000):
        slab = "10%"
    elif (salary * 12 <= 600000):
        slab = "20%"
    elif (salary * 12 <= 1000000):
        slab = "30%"
    elif (salary * 12 > 2000000):
        slab = "36%"

    print(f'''
        Staff {name} with PAN {pan} fall under {slab} Tax slab.
        {name} with (PAN {pan}) has to pay tax to government is [Rs.]= {taxamount}
    ''')


def staff_info():
    staffname = []
    staffaddress = []
    staffpan = []
    stafffy = []
    staffincome = []
    married = []
    taxamount = []
    staffno = int(input("Enter the number of staff you want to provide data: "))
    for i in range(staffno):
        print(f"Enter for the {i + 1} Staff Information: ")
        staffname.append(input(f"Enter Staff Name[{i + 1}]: "))
        staffaddress.append(input(f"Enter Address[{i + 1}]: "))
        staffpan.append(input(f"Enter PAN No[{i + 1}]: "))
        married.append(input(f"Enter 'Y' for Married and 'N' for Unmarried Status[{i + 1}]: "))
        stafffy.append(input(f"Enter FY[{i + 1}]: "))
        staffincome.append(int(input(f"Enter Staff per month income[Rs.][{i + 1}]: ")))
        if (married == "Y"):
            taxamount.append(calculate_tax_of_married(staffincome[i]))
        else:
            taxamount.append(calculate_tax_of_staff(staffincome[i]))
    for i in range(staffno):
        display_staff_info(staffname[i], staffaddress[i], staffpan[i], staffincome[i], taxamount[i],
        stafffy[i],married[i])
        f = open("C:/Users/ASUS/OneDrive/Desktop/python", "a")
        f.write("Staff name: " + staffname[i] + ", ")
        f.write("Staff address: " + staffaddress[i] + ", ")
        f.write("Staff PAN: " + staffpan[i] + ", ")
        f.write("For Year: " + stafffy[i] + ", ")
        f.write("Married Status: " + married[i] + ",")
        f.write("Staff income: " + str(staffincome[i]) + ", ")
        f.write("Tax amount: " + str(taxamount[i]) + "\n")
        f.close()
staff_info()