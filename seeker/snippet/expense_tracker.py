#date: 2025-07-21T17:10:35Z
#url: https://api.github.com/gists/1ac31992a190fd7a48fab71db960f064
#owner: https://api.github.com/users/dhvanityadav

from expense_track_func import *

print("PERSONAL EXPENSE TRACKER".center(50, "*"))


def main():

    initialize_file()
    while True:
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Filter Expenses")
        print("4. Delete Expense")
        print("5. Monthly Summary")
        print("6. Exit")

        choice = int(input("Select an option (1-6): "))

        if (choice == 1):
            date = input("Enter date (YYYY-MM-DD): ")
            amount = input("Enter amount: ")
            category = input("Enter category: ")
            description = input("Enter description: ")
            add_expense(date, amount, category, description)

        elif (choice == 2):
            view_expenses()

        elif (choice == 3):
            filter_by = input("Filter by (Date/Category): ")
            filter_value = input(f"Enter {filter_by}: ")
            filter_expenses(filter_by, filter_value)

        elif (choice == 4):
            date = input("Enter date (YYYY-MM-DD): ")
            amount = input("Enter amount: ")
            category = input("Enter category: ")
            description = input("Enter description: ")
            delete_expense(date, amount, category, description)
            print()

        elif (choice == 5):
            monthly_summery()

        elif (choice == 6):
            print("Exiting the program. Goodbye!")

            break

        else:
            print("Invalid Option. Please try again.")


if __name__ == "__main__":
    main()




