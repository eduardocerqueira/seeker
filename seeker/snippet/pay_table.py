#date: 2023-12-21T16:42:43Z
#url: https://api.github.com/gists/a323d2c41619145b06cbf45a53759982
#owner: https://api.github.com/users/bbq-bean

def get_pay_table(start, increase, years):
    # this will grow
    current_salary = start
    # increase is already of type float result will be float
    modifier = increase / 100
    #print(modifier)

    print("\nYear Salary\n------------")
    # if someone inputs 10, they expect a table with 10 rows
    for i in range(1, years + 1):
        # base case
        if i == 1:
            print(f"{i} {current_salary}")
        
        else:
            current_salary = round(current_salary + (current_salary * modifier), 2)
            print(f"{i} {current_salary}") 
        

def clean_input(input):
    """check if input is numerical-ish"""
    try:
        float(input)
    
    except ValueError:
        return False
    
    return True


if __name__ == "__main__":
    # get starting salary, validate input
    while True:
        s = input("Enter starting pay: ")
        
        if clean_input(s):
            s = float(s)
            break

        print("bad input. Enter whole numbers or decimals, eg: 10000 or 10000.00")

    # get percent increase
    while True:
        i = input("Enter yearly increase: ")

        if clean_input(i):
            i = float(i)
            break

        print("bad input. Enter whole numbers, dont include '%' eg: 2 or 3")
    
    # get years
    while True:
        y = input("Enter number of years to calculate: ")
        
        if clean_input(y):
            # this will still pass if they put "2.5" years, but dont care, 
            # it will just output 2 years instead as partial year is not a feature
            y = int(y)
            break

        print("bad input. Enter whole numbers, eg: 5 or 20")
    

    get_pay_table(s, i, y)
