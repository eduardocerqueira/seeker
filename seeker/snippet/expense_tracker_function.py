#date: 2025-07-21T17:10:35Z
#url: https://api.github.com/gists/1ac31992a190fd7a48fab71db960f064
#owner: https://api.github.com/users/dhvanityadav

import os
import datetime

def initialize_file():
    if os.path.exists("Expenses.txt"):
        with open("Expenses.txt", "w") as f:
            f.write("Data, Amount, Category, Description\n")


def add_expense(data, amount, category, description):
    with open("Expenses.txt", "a") as f:
        f.write(f"{data}, {amount}, {category}, {description}\n")
    print("Expense added successfully.")

def view_expenses():
    with open("Expenses.txt", "r") as f:
        expenses = f.readlines()
        print(expenses[0])
        for expense in expenses[1:]:
            print(expense)

    # if (len(expenses) <= 1):
    #     print("No expenses recorded yet.")

def filter_expenses(filter_by, filter_value):
    with open("Expenses.txt", "r") as f:
        expenses  = f.readlines()
        print(expenses[0])
        for expense in expenses [1:]:
            data = expense.split(", ")
            if filter_by == "Date" and filter_value in data[0]:
                print(expense)
            elif filter_by == "Category" and filter_value == data[2]:
                print(expense)

def delete_expense(data, amount, category, description):
    expenses = []
    with open("Expenses.txt", "r") as f:
        expenses = f.readlines()
    with open("Expenses.txt", "w") as f:
        for expense in expenses:
            if expense != f"{data}, {amount}, {category}, {description}\n":
                f.write(expense)
    print("Expense deleted successfully.")

def monthly_summery():
    current_month = datetime.datetime.now().strftime("%Y-%m")
    total_amount = 0.0
    category_expense = {}

    with open("Expenses.txt", "r") as f:
        expenses = f.readlines()
        for expense in expenses:
            data = expense.strip().split(", ")
            if data[0].startswith(current_month):
                amount = float(data[1])
                category = data[2]
                total_amount += amount
            if category in category_expense:
                category_expense[category] += amount
            else:
                category_expense[category] = amount

    print(f"Total expenses for {current_month}: {total_amount}")

    for category, amount in category_expense.items():
        print(f"Category: {category}, Amount: {amount}")



    