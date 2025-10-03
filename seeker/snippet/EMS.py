#date: 2025-10-03T16:48:54Z
#url: https://api.github.com/gists/cdfc872c87ee494644d5c39212c7fe0f
#owner: https://api.github.com/users/Bhav-creator452

#Employee Management System 

employees={
    101:{"Name": "Sukhmani","Age":25,"Department":"HR","Salary":50000},
    102:{"Name": "Harshdeep","Age":26,"Department":"Finance","Salary":55000},
    103:{"Name": "Bhavdeep","Age":25,"Department":"IT","Salary":60000}
}

def add_employee():
    """Add a New Employee """
    while True:
        try:
            emp_id=int(input("Enter Employee ID:"))
            if emp_id in employees:
                print("Employee ID already exists . Please enter a new one.")
                continue
            break
        except ValueError:
            print("❌ Invalid Input! Employee ID must be a number.")

    name=input("Enter Employee Name:")
    age=int(input("Enter Employee Age:"))
    department=input("Enter Employee Department:")
    salary=float(input("Enter Employee Salary:"))

    employees[emp_id]={
    "Name":name,
    "Age":age,
    "Department":department,
    "Salary":salary
}
    print(f"Employee: {name} added successfully!✅ ")

#add_employee()


def view_employee():
    """Display employees in a table-like format"""
    if not employees:
        print("sorry! no employees availaible.\n")
        return  #this is only true when "employees is empty"
    
    print("---Employees list---\n")
    print(f"{'ID':<10}{'Name':<15}{'Department':<15}{'Salary':<10}")
    print("-"*60)
    for emp_id,details in employees.items():
        print(f"{emp_id:<10}{details['Name']:<15}{details['Age']:<10}{details['Department']:<15}{details['Salary']:<10}")
    print()

#view_employee()


def search_employee():
    """Search for an employee by their ID"""
    try:
        emp_id=int(input("Enter Employee ID you want to search:"))
        if emp_id in employees:
            emp=employees[emp_id]
            print("----Employee Found----\n")
            print(f"ID:{emp_id}")
            print(f"Name:{emp["Name"]}")
            print(f"Age:{emp["Age"]}")
            print(f"Department:{emp["Department"]}")
            print(f"Salary:{emp["Salary"]}\n")
        else:
            print("❌ Employee not found!")
    except ValueError:
        print("🚫 Invalid Input! Employee ID must be a number.")

#search_employee()

def main_menu():
    """Display the main menu and handle user input"""
    while True:
        print("====Employee Management System====")
        print("1. Add new Employee")
        print("2. View list of all Employees")
        print("3. Search for an Employee")
        print("4. Exit")

        choice=input("Enter yout choice (1-4):")

        if choice=="1":
            add_employee()
        elif choice=="2":
            view_employee()
        elif choice=="3":
            search_employee()
        elif choice=="4":
            print("Thank you for using our EMS!")
            break
        else:
            print("Invalid Input! PLease try again!")
        
main_menu()





