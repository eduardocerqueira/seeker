#date: 2024-06-13T17:07:05Z
#url: https://api.github.com/gists/b93f6da116230d0254ff78e5d988450c
#owner: https://api.github.com/users/DharunKumar04

import sqlite3
from prettytable import PrettyTable
from termcolor import cprint

def initialize_database():
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY,
                company_name TEXT,
                date TEXT,
                shares INTEGER,
                inr_per_share REAL,
                platform TEXT
            )
        ''')

def store_stock_data(company_name, date, shares, inr_per_share, platform):
    try:
        shares = int(shares)
        inr_per_share = float(inr_per_share)
    except ValueError:
        print("Invalid input. Shares should be an integer, and INR per share should be a float.")
        return

    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        try:
            cursor.execute('INSERT INTO stocks (company_name, date, shares, inr_per_share, platform) VALUES (?, ?, ?, ?, ?)',
                           (company_name, date, shares, inr_per_share, platform))
            connection.commit()
            print("Stock data added successfully.")
        except Exception as e:
            print(f"Error: {e}")

def list_all_stock_details():
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM stocks ORDER BY company_name ASC, date ASC')
        stocks = cursor.fetchall()

        if stocks:
            table = PrettyTable()
            table.field_names = ["ID", "Company Name", "Date", "No of Shares", "INR per Share", "Platform"]

            for stock in stocks:
                table.add_row([stock[0], stock[1], stock[2], stock[3], stock[4], stock[5]])

            table.align = 'l'
            table.border = True
            print("\n" + str(table))
            
            return table
        else:
            print("No stock records found.")
            return None
            
def list_all_company_names():
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        cursor.execute('SELECT DISTINCT company_name FROM stocks ORDER BY company_name ASC')
        companies = cursor.fetchall()

        if companies:
            print("\nList of all company names:")
            for company in companies:
                print(company[0])
        else:
            print("No company names found.")

def list_all_stocks_by_date(search_date):
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM stocks WHERE date = ? ORDER BY date ASC, company_name ASC', (search_date,))
        stocks = cursor.fetchall()

        if stocks:
            table = PrettyTable()
            table.field_names = ["Date", "Company Name", "No of Shares", "INR per Share", "Platform"]

            for stock in stocks:
                table.add_row([stock[2], stock[1], stock[3], stock[4], stock[5]])

            table.align = 'l'
            table.border = True
            print("\n" + str(table))
            
            return table
        else:
            print(f"No stock records found for date {search_date}.")
            return None

def delete_stock_data_by_date(search_date):
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        try:
            cursor.execute('DELETE FROM stocks WHERE date = ?', (search_date,))
            connection.commit()
            print(f"All stock data for date {search_date} deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")

def list_stocks_by_name(company_name):
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM stocks WHERE company_name = ? ORDER BY company_name ASC, date ASC', (company_name,))
        stocks = cursor.fetchall()

        if stocks:
            table = PrettyTable()
            table.field_names = ["Company Name", "Date", "No of Shares", "INR per Share", "Platform"]

            for stock in stocks:
                table.add_row([stock[1], stock[2], stock[3], stock[4], stock[5]])

            table.align = 'l'
            table.border = True
            print("\n" + str(table))
            
            return table
        else:
            print(f"No stock records found for company name {company_name}.")
            return None

def delete_stock_data_by_name(company_name):
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        try:
            cursor.execute('DELETE FROM stocks WHERE company_name = ?', (company_name,))
            connection.commit()
            print(f"All stock data for company name {company_name} deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")

def get_stock_id(company_name, date, shares, inr_per_share, platform):
    try:
        shares = int(shares)
        inr_per_share = float(inr_per_share)
    except ValueError:
        print("Invalid input. Shares should be an integer, and INR per share should be a float.")
        return

    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        try:
            cursor.execute('SELECT id FROM stocks WHERE company_name = ? AND date = ? AND shares = ? AND inr_per_share = ? AND platform = ?',
                           (company_name, date, shares, inr_per_share, platform))
            stock_id = cursor.fetchone()

            if stock_id:
                print(f"ID for the provided stock details: {stock_id[0]}")
            else:
                print("No matching stock details found.")
        except Exception as e:
            print(f"Error: {e}")

def delete_stock_data_by_id(stock_id):
    try:
        stock_id = int(stock_id)
    except ValueError:
        print("Invalid input. ID should be an integer.")
        return

    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        try:
            cursor.execute('DELETE FROM stocks WHERE id = ?', (stock_id,))
            connection.commit()
            print(f"Stock data with ID {stock_id} deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")

def list_stocks_by_platform(platform):
    with sqlite3.connect('stock_data.db') as connection:
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM stocks WHERE platform = ? ORDER BY platform ASC, company_name ASC, date ASC', (platform,))
        stocks = cursor.fetchall()

        if stocks:
            table = PrettyTable()
            table.field_names = ["Company Name", "Date", "No of Shares", "INR per Share", "Platform"]

            for stock in stocks:
                table.add_row([stock[1], stock[2], stock[3], stock[4], stock[5]])

            table.align = 'l'
            table.border = True
            print("\n" + str(table))

            return table
        else:
            print(f"No stock records found for platform {platform}.")
            return None

def copy_stock_data_to_file(table):
    if table:
        with open('data.txt', 'w') as file:
            file.write(table.get_string())
        
        print("Stock data copied to data.txt successfully.")
    else:
        print("No stock records found.")

if __name__ == '__main__':
    initialize_database()
    cprint('STOCK MANAGER'.center(50), 'cyan', attrs=['bold'])
    print("\nPress 0 to add stock data")
    print("\n")
    print("Press 1 to list all company names")
    print("Press 2 to display all stock details")
    print("\n")
    print("Press 3 to list all stock details for a particular date")
    print("Press 4 to delete all stock entries for a particular date")
    print("\n")
    print("Press 5 to list all stock details for a particular company name")
    print("Press 6 to delete stock data by company name")
    print("\n")
    print("Press 7 to get the ID of an entry based on user-provided information")
    print("Press 8 to delete an entry by providing the ID")
    print("\n")
    print("Press 9 to list all stock details for a particular platform")
    print("\n")
    print("Press 10 to copy all stock details to data.txt file")

    while True:
        choice = input("\nEnter your choice (press Enter to exit): ")

        if choice == '':
            print(" Exiting the loop.")
            break
        elif choice == '0':
            company_name = input(" Enter company name: ").strip()
            date = input(" Enter date (dd/mm/yyyy): ").strip()
            shares = input(" Enter number of shares: ").strip()
            inr_per_share = input(" Enter INR per share: ").strip()
            platform = input(" Enter platform: ").strip()

            store_stock_data(company_name, date, shares, inr_per_share, platform)
        elif choice == '1':
            list_all_company_names()
        elif choice == '2':
            list_all_stock_details()
        elif choice == '3':
            search_date = input(" Enter date (dd/mm/yyyy) to list all stock details: ").strip()
            table = list_all_stocks_by_date(search_date)
        elif choice == '4':
            search_date = input(" Enter date (dd/mm/yyyy) to delete all stock entries: ").strip()
            delete_stock_data_by_date(search_date)
        elif choice == '5':
            company_name = input(" Enter company name to list all stock details: ").strip()
            table = list_stocks_by_name(company_name)
        elif choice == '6':
            company_name = input(" Enter company name to delete: ").strip()
            delete_stock_data_by_name(company_name)
        elif choice == '7':
            company_name = input(" Enter company name: ").strip()
            date = input(" Enter date (dd/mm/yyyy): ").strip()
            shares = input(" Enter number of shares: ").strip()
            inr_per_share = input(" Enter INR per share: ").strip()
            platform = input(" Enter platform: ").strip()

            get_stock_id(company_name, date, shares, inr_per_share, platform)
        elif choice == '8':
            stock_id = input(" Enter ID to delete the entry: ").strip()

            try:
                stock_id = int(stock_id)
            except ValueError:
                print("Invalid input. ID should be an integer.")
                continue

            delete_stock_data_by_id(stock_id)
        elif choice == '9':
            platform = input(" Enter platform to list all stock details: ").strip()
            table = list_stocks_by_platform(platform)
        elif choice == '10':
            copy_stock_data_to_file(table)
        elif choice.strip():
            print(" Invalid choice. Please enter a valid option.")