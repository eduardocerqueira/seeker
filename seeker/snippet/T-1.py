#date: 2023-01-16T16:44:44Z
#url: https://api.github.com/gists/c948f5399dd9e3325fe8d4a298054927
#owner: https://api.github.com/users/Suraj542005

import csv


def ADDITION():
    with open(r"C:\Users\Suraj\OneDrive\Documents\csv_files_Suraj\Employ_T1.csv", 'a', newline='') as file_object:
        file_writer = csv.writer(file_object)
        c = 'y'
        while c == 'y':
            empid = int(input("Enter the Employ ID :"))
            name = input("Enter the Name :")
            salary = int(input("Enter the Salary :"))
            data = [empid, name, salary]
            file_writer.writerow(data)
            print("Data successfully stored.....")
            c = input("Enter 'y' to continue :")


def COUNT():
    with open(r"C:\Users\Suraj\OneDrive\Documents\csv_files_Suraj\Employ_T1.csv", 'r', newline='') as file_object:
        file_reader = csv.reader(file_object)
        print("file_reader :", file_reader)
        for i in file_reader:
            print(i)


ADDITION()
COUNT()
