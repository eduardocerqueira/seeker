#date: 2023-10-10T17:08:15Z
#url: https://api.github.com/gists/5d6ce720c820119d7a3a5b4d41ec53d6
#owner: https://api.github.com/users/MariamGuliashvili

def get_month_name(month_number):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    if month_number >= 1 and month_number <= 12:
        return months[month_number - 1]
    else:
        return "Invalid month number. Please enter a number between 1 and 12."

def main():
    try:
        month_number = int(input("Enter a number between 1 and 12: "))
        month_name = get_month_name(month_number)
        print(f"The corresponding month is: {month_name}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

if __name__ == "__main__":
    main()
