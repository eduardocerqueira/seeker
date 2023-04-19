#date: 2023-04-19T16:51:29Z
#url: https://api.github.com/gists/20317868c79e2e2d88d2c95298843bc9
#owner: https://api.github.com/users/t0023656

import json
from tkinter import *
from tkinter import messagebox
from random import choice, randint, shuffle
import pyperclip


# ---------------------------- PASSWORD GENERATOR ------------------------------- #

# Password Generator Project
 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********") "**********": "**********"
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

    password_letters = "**********"
    password_symbols = "**********"
    password_numbers = "**********"

    password_list = "**********"
    shuffle(password_list)

    password = "**********"
    password_entry.insert(0, password)
    pyperclip.copy(password)


# ---------------------------- SAVE PASSWORD ------------------------------- #
def save():
    website = website_entry.get()
    email = email_entry.get()
    password = "**********"
    new_data = {website: {
        "email": email,
        "password": "**********"
    }}

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"w "**********"e "**********"b "**********"s "**********"i "**********"t "**********"e "**********") "**********"  "**********"= "**********"= "**********"  "**********"0 "**********"  "**********"o "**********"r "**********"  "**********"l "**********"e "**********"n "**********"( "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********"  "**********"= "**********"= "**********"  "**********"0 "**********": "**********"
        messagebox.showinfo(title="Oops", message="Please make sure you haven't left any fields empty.")
    else:
        is_ok = messagebox.askokcancel(title=website, message=f"These are the details entered: \nEmail: {email} "
                                                              f"\nPassword: "**********"
        if is_ok:
            try:
                with open("data.json", "r") as data_file:
                    data = json.load(data_file)
            except FileNotFoundError:
                with open("data.json", "w") as data_file:
                    json.dump(new_data, data_file, indent=4)
            else:
                data.update(new_data)

                with open("data.json", "w") as data_file:
                    json.dump(data, data_file, indent=4)
            finally:
                website_entry.delete(0, END)
                password_entry.delete(0, END)


def search():
    website = website_entry.get()
    try:
        with open("data.json", "r") as data_file:
            data = json.load(data_file)
            messagebox.showinfo(title=website, message=f"Email: {data[website]['email']}\n"
                                                       f"Password: "**********"
    except FileNotFoundError:
        messagebox.showinfo(title="!", message="沒有存檔")
    except KeyError:
        messagebox.showinfo(title="!", message="沒有紀錄")


# ---------------------------- UI SETUP ------------------------------- #

window = Tk()
window.title("Password Manager")
window.config(padx=50, pady=50)

canvas = Canvas(height=200, width=200)
logo_img = PhotoImage(file="logo.png")
canvas.create_image(100, 100, image=logo_img)
canvas.grid(row=0, column=1)

# Labels
website_label = Label(text="Website:")
website_label.grid(row=1, column=0)
email_label = Label(text="Email/Username:")
email_label.grid(row=2, column=0)
password_label = Label(text="Password: "**********"
password_label.grid(row= "**********"=0)

# Entries
website_entry = Entry(width=21)
website_entry.grid(row=1, column=1)
website_entry.focus()
email_entry = Entry(width=35)
email_entry.grid(row=2, column=1, columnspan=2)
email_entry.insert(0, "angela@gmail.com")
password_entry = "**********"=21)
password_entry.grid(row= "**********"=1)

# Buttons
generate_password_button = "**********"="Generate Password", command=generate_password)
generate_password_button.grid(row= "**********"=2)
add_button = Button(text="Add", width=36, command=save)
add_button.grid(row=4, column=1, columnspan=2)
search_button = Button(text="Search",width=15, command=search)
search_button.grid(row=1, column=2)

window.mainloop()
