#date: 2023-05-29T16:47:26Z
#url: https://api.github.com/gists/bdea4dc37d75fa8819884c4699fa13ba
#owner: https://api.github.com/users/JohanMacci

import tkinter as tk
import tkinter.font as font
import tkinter.ttk as ttk
from tkinter import filedialog
import sys
import csv
import os
import pysftp
import warnings
import random
from datetime import datetime
import pycountry
import io
from io import TextIOWrapper
import mimetypes
import platform
import xml.etree.ElementTree as ET
# Import environment variables
from dotenv import load_dotenv
from tkcalendar import DateEntry
import stat
import tkinter.messagebox
from functools import partial


global generate_button
generate_button = None


if platform.system() == "Windows":
    # Adjustments for Windows
    pass
elif platform.system() == "Linux":
    # Adjustments for Linux
    pass
elif platform.system() == "Darwin":  # macOS
    # Adjustments for macOS
    pass




load_dotenv()

# Ignore the UserWarning related to host keys
warnings.filterwarnings("ignore", category=UserWarning)

file_path = os.path.join("path", "to", "file.txt")

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

current_year = datetime.now().year

# Create a list of years starting from the current year + 2 and descending
years = list(range(current_year + 2, current_year - 98, -1))
countries = [country.name for country in pycountry.countries]

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# SFTP server credentials and CSV file
sftp_server = 'digitalhubftp.fifa.org'
sftp_user = 'CSVadmin'
sftp_pass = 'uo!)KF8WtpTxy0'


def get_sftp_folders():
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"p "**********"y "**********"s "**********"f "**********"t "**********"p "**********". "**********"C "**********"o "**********"n "**********"n "**********"e "**********"c "**********"t "**********"i "**********"o "**********"n "**********"( "**********"s "**********"f "**********"t "**********"p "**********"_ "**********"s "**********"e "**********"r "**********"v "**********"e "**********"r "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"= "**********"s "**********"f "**********"t "**********"p "**********"_ "**********"u "**********"s "**********"e "**********"r "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"= "**********"s "**********"f "**********"t "**********"p "**********"_ "**********"p "**********"a "**********"s "**********"s "**********", "**********"  "**********"c "**********"n "**********"o "**********"p "**********"t "**********"s "**********"= "**********"c "**********"n "**********"o "**********"p "**********"t "**********"s "**********") "**********"  "**********"a "**********"s "**********"  "**********"s "**********"f "**********"t "**********"p "**********": "**********"
        sftp.chdir('/')
        folders = [entry.filename for entry in sftp.listdir_attr() if entry.st_mode & stat.S_IFDIR]
        print(f"Folders found: {folders}")

        if not folders:
            return [], [], []

        remote_csv_files = []  # Store the found CSV files in a list
        column_names = None
        for folder in folders:  # Loop through all folders
            sftp.chdir('/')  # Reset the working directory back to the root directory
            sftp.chdir(folder)
            for file in sftp.listdir():
                if file.endswith('.csv'):
                    remote_csv_files.append((folder, file))  # Add the folder and CSV file as a tuple to the list
                    print(f"CSV file found in folder {folder}: {file}")  # Print the found CSV file
                    
                    # Fetch column names from the first CSV file found
                    if column_names is None:
                        with sftp.open(file, mode='r') as f:
                            wrapped_file = io.TextIOWrapper(f, encoding='cp1252')
                            reader = csv.reader(wrapped_file, delimiter=';')
                            column_names = next(reader)  # Read the first row as column names
  # Read the first row as column names

    return folders, remote_csv_files, column_names  # Return the list of found CSV files and column names



def find_transmission_ref_index(column_names):
    try:
        transmission_ref_index = column_names.index("TransmissionReference")
    except ValueError:
        transmission_ref_index = -1
    return transmission_ref_index


def generate_random_value():
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    random_number = random.randint(10, 99)
    random_letter = random.choice("abcdefghijklmnopqrstuvwxyz")
    random_value = f"{date_str}{random_number}{random_letter}"

    if "TransmissionReference" in column_names:
        transmission_ref_index = column_names.index("TransmissionReference")
        if transmission_ref_index < len(entry_vars):
            entry_vars[transmission_ref_index].set(random_value)

    if "JobID" in column_names:
        job_id_index = column_names.index("JobID")
        if job_id_index < len(entry_vars):
            entry_vars[job_id_index].set(random_value)

    if "FIFA_President_Event" in column_names:
        president_event_index = column_names.index("FIFA_President_Event")
        if president_event_index < len(entry_vars):
            entry_vars[president_event_index].set(random_value)



def save_data():
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    cnopts.auto_add_key = True

    with pysftp.Connection(
        sftp_server, username= "**********"=sftp_pass, cnopts=cnopts
    ) as sftp:
        data = [entry_var.get() if i < len(entry_vars) else "" for i, entry_var in enumerate(entry_vars)]

        # Copy Transmission Reference value to JobID
        transmission_ref_index = column_names.index("TransmissionReference")
        job_id_index = column_names.index("JobID")
        data[job_id_index] = data[transmission_ref_index]

        folder = selected_folder.get()
        # Fetch the list of files in the selected folder
        file_list = sftp.listdir(folder)

        # Find the first CSV file in the folder
        remote_csv_file = None
        for file in file_list:
            if file.endswith('.csv'):
                remote_csv_file = f"{folder}/{file}"
                break

        if remote_csv_file is None:
            tk.messagebox.showerror("Error", "No CSV file found in the selected folder.")
            return

        local_csv_file = remote_csv_file.split("/")[-1]

        # Download the existing CSV file from the SFTP server
        with sftp.open(remote_csv_file, mode='rb') as f:
            content = f.read()
            try:
                content_str = content.decode('cp1252')
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError: {e}")
                start = e.start
                end = e.end if e.end - e.start < 10 else e.start + 10  # limit to 10 characters
                print(f"Content around problematic area: {content[start-10:end+10]}")
                return

            csv_data = list(csv.reader(content_str.splitlines(), delimiter=';'))

        # Append the new data to the dataset
        csv_data.append(data)

        # Convert data back to bytes with the appropriate encoding
        csv_bytes = io.StringIO()
        writer = csv.writer(csv_bytes, delimiter=';')
        writer.writerows(csv_data)
        csv_bytes = csv_bytes.getvalue().encode('cp1252')

        # Upload the updated CSV file to the SFTP server
        with sftp.open(remote_csv_file, mode='wb') as f:
            f.write(csv_bytes)  # Write the updated CSV data to the file

    transmission_ref_index = find_transmission_ref_index(column_names)
    if transmission_ref_index >= 0:
        data[transmission_ref_index] = data[job_id_index]

        # Clear input boxes
        for entry_var in entry_vars:
            entry_var.set("")

def refresh_gui(transmission_ref_index):
    global entry_vars, dropdown_options, column_names, save_button

    # Destroy existing input fields
    for entry in app.grid_slaves():
        if isinstance(entry, (tk.Entry, ttk.Combobox, tk.Label, tk.Button)):
            entry.destroy()

    # Clear existing entry_vars
    entry_vars.clear()

    # Create new input fields based on the updated column_names
    for i, column_name in enumerate(column_names):
        label = tk.Label(app, text=column_name)
        label.grid(row=i, column=0)

        if column_name in dropdown_options:
            var = tk.StringVar(app)
            var.set(dropdown_options[column_name][0])
            entry = ttk.Combobox(app, textvariable=var, values=dropdown_options[column_name], height=10, width=30)
            entry.grid(row=i, column=1)

        else:
            var = tk.StringVar(app)

            if column_name == "TransmissionReference":
                entry = tk.Entry(app, textvariable=var, width=30)
                entry.grid(row=i, column=1)

                # Create the generate_button here
                generate_button = tk.Button(app, text="Generate", command=generate_random_value)
                generate_button.grid(row=i, column=2, padx=10)

            elif column_name == "Event Start Date":
                entry = DateEntry(app, textvariable=var, date_pattern='dd/mm/Y', width=30)
                entry.grid(row=i, column=1)

            elif column_name == "Country":
                country_entry = tk.Entry(app, textvariable=var, width=30)
                country_entry.bind('<KeyRelease>', partial(autofill, entry=country_entry))


                entry = country_entry
                entry.grid(row=i, column=1, columnspan=1)
            else:
                entry = tk.Entry(app, textvariable=var, width=30)
                entry.grid(row=i, column=1)

        entry_vars.append(var)

    save_button.grid(row=len(column_names), column=1, pady=10)  # Move the save button

    global folder_selector
    folder_selector.destroy()
    folder_selector = ttk.Combobox(app, textvariable=selected_folder, values=folders, height=10, width=30)
    folder_selector.grid(row=0, column=3, padx=10)
    folder_selector.bind("<<ComboboxSelected>>", folder_changed)



def folder_changed(*args):
    global column_names
    current_folder = selected_folder.get()
    csv_file = None
    for folder, file in csv_files:
        if folder == current_folder:
            csv_file = file
            break

    if csv_file:
        print(f"Current folder: {current_folder}, CSV file: {csv_file}")
     "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"p "**********"y "**********"s "**********"f "**********"t "**********"p "**********". "**********"C "**********"o "**********"n "**********"n "**********"e "**********"c "**********"t "**********"i "**********"o "**********"n "**********"( "**********"s "**********"f "**********"t "**********"p "**********"_ "**********"s "**********"e "**********"r "**********"v "**********"e "**********"r "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"= "**********"s "**********"f "**********"t "**********"p "**********"_ "**********"u "**********"s "**********"e "**********"r "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"= "**********"s "**********"f "**********"t "**********"p "**********"_ "**********"p "**********"a "**********"s "**********"s "**********", "**********"  "**********"c "**********"n "**********"o "**********"p "**********"t "**********"s "**********"= "**********"c "**********"n "**********"o "**********"p "**********"t "**********"s "**********") "**********"  "**********"a "**********"s "**********"  "**********"s "**********"f "**********"t "**********"p "**********": "**********"
            with sftp.open(f"{current_folder}/{csv_file}", mode='r') as f:
                wrapped_file = io.TextIOWrapper(f, encoding='cp1252')
                reader = csv.reader(wrapped_file, delimiter=';')
                column_names = next(reader)  # Read the first row as column names
                print(f"Updated column names: {column_names}")

    

    else:
        print(f"Current folder: {current_folder}, No CSV file found")

      # Pass the transmission_ref_index as an argument

    # Add this line before the first call to the refresh_gui function
    global save_button
    save_button = tk.Button(app, text="Save", command=save_data)

    transmission_ref_index = find_transmission_ref_index(column_names)
    refresh_gui(transmission_ref_index)


folders, csv_files, column_names = get_sftp_folders()
print(folders)
print(csv_files)
print(column_names)



# GUI setup
app = tk.Tk()
app.title("CSV Data Entry")

# Disable window resizing
app.resizable(False, False)

# Disable the maximize button
app.protocol("WM_DELETE_WINDOW", app.destroy)

entry_vars = []
generate_button = None

default_font = font.nametofont("TkDefaultFont")
default_font.actual()  # Returns a dictionary containing the default font properties
default_font.configure(size=12)  # Set the size you want

selected_folder = tk.StringVar(app)
selected_folder.set(folders[0])  # Set the first folder as the default selection

folder_selector = ttk.Combobox(app, textvariable=selected_folder, values=folders, height=10, width=30)
folder_selector.grid(row=0, column=3, padx=10)
folder_selector.bind("<<ComboboxSelected>>", folder_changed)



global suggestions_listbox
suggestions_listbox = None



# Dropdown menu options
dropdown_options = {
    "FileType": ["Images", "Graphics_logos", "Documents", "Audio", "Video"],
    "Digital_Rights": ["FIFA_Internal", "FIFAeInternal", "External_Users","Commercial_Partners_Material", "Editorial", "Confidential_Admins_Only","FIFAcom", "Licensed", "Embargo","MediaHub", "ToBeReviewed", "Uncategorized", "Brand"],
    "FIFA_President_Year": [""] + years,
    "FIFA_President_Month": [""] + months,
    "FileContent": ["", "Competition", "FIFAMuseum", "FIFATV", "FIFAPlus", "Guidelines", "KnowledgeCapture", "Non-Competition", "Organisational", "QatarLLC"],
    "SubContent": "**********"
    

}



def select_suggestion(event):
    if suggestions_listbox:
        selection = suggestions_listbox.curselection()
        if selection:
            country_entry.delete(0, tk.END)
            country_entry.insert(0, suggestions_listbox.get(selection))
            suggestions_popup.withdraw()

def autofill(event, entry):  # Add the entry argument here

    global suggestions_listbox
    global suggestions_popup

    # Destroy the suggestions_listbox when the Enter key is pressed
    if event.keysym == 'Return':
        if suggestions_listbox:
            entry.delete(0, tk.END)
            entry.insert(0, suggestions_listbox.get(tk.ACTIVE))

            suggestions_listbox.destroy()
            suggestions_popup.destroy()
            suggestions_listbox = None
            suggestions_popup = None
        return

    # Handle keyboard navigation
    if event.keysym in ('Up', 'Down'):
        if suggestions_listbox:
            index = suggestions_listbox.curselection()
            if index:
                index = index[0]
            else:
                index = 0
            if event.keysym == 'Up':
                index = max(0, index - 1)
            else:
                index = min(len(suggestions_listbox.get(0, tk.END)) - 1, index + 1)
            suggestions_listbox.selection_clear(0, tk.END)
            suggestions_listbox.selection_set(index)
            suggestions_listbox.activate(index)
        return

    # Filter and update suggestions
    current_value = entry.get()

    if current_value:
        suggestions = [c for c in countries if c.lower().startswith(current_value.lower())]

        if not suggestions_listbox:
            suggestions_popup = tk.Toplevel(app)
            suggestions_popup.overrideredirect(True)
            suggestions_popup.geometry("+%s+%s" % (entry.winfo_rootx(), entry.winfo_rooty() + entry.winfo_height()))

            suggestions_listbox = tk.Listbox(suggestions_popup, width=entry["width"], height=5)

            suggestions_listbox.pack()

        suggestions_listbox.delete(0, tk.END)
        for s in suggestions:
            suggestions_listbox.insert(tk.END, s)

        suggestions_listbox.bind('<Button-1>', select_suggestion)

    else:
        if suggestions_listbox:
            suggestions_popup.withdraw()




#default values;
copyright_today = datetime.now()
copyright_year = copyright_today.strftime("%Y")
copyright_name = "FIFA"
copyright_default = f"{copyright_year} {copyright_name}"
credit_default = "FIFA"
source_default = "FIFA"

save_button = tk.Button(app, text="Save to CSV", command=save_data)
save_button.grid(row=len(column_names), column=1, pady=10)

app.mainloop()