#date: 2023-11-20T16:27:03Z
#url: https://api.github.com/gists/c7c0b1a3f386a1a934edfdd8d5f03304
#owner: https://api.github.com/users/AurelDeveloper

import json

class GradeBook:
    def __init__(self):
        self.subjects = {}

    def add_grade(self, subject, grade):
        if 1 <= grade <= 6:
            if subject in self.subjects:
                self.subjects[subject].append(grade)
            else:
                self.subjects[subject] = [grade]
            print(f"\033[92m✅ Grade {grade} added to subject {subject}.\033[0m")  
            self.save_data("gradebook.json")
        else:
            print("\033[91m❌ Invalid grade. Please enter a grade between 1 and 6.\033[0m")  

    def add_subject(self, subject):
        if subject not in self.subjects:
            self.subjects[subject] = []
            print(f"\033[92m✅ Subject {subject} added.\033[0m")  
            self.save_data("gradebook.json")  
        else:
            print(f"\033[93m⚠️ Subject {subject} already exists.\033[0m")  

    def delete_subject(self, subject):
        if subject in self.subjects:
            del self.subjects[subject]
            print(f"\033[92m✅ Subject {subject} deleted.\033[0m")
            self.save_data("gradebook.json")
        else:
            print(f"\033[91m❌ Subject {subject} does not exist.\033[0m")  

    def delete_grade(self, subject, grade):
        if subject in self.subjects and grade in self.subjects[subject]:
            self.subjects[subject].remove(grade)
            print(f"\033[92m✅ Grade {grade} deleted from subject {subject}.\033[0m")  
            self.save_data("gradebook.json") 
        else:
            print(f"\033[91m❌ Subject {subject} or grade {grade} not found.\033[0m") 

    def show_averages(self):
        for subject, grades in self.subjects.items():
            if grades:
                average = sum(grades) / len(grades)
                color = "\033[94m"
                print(f"{color}📘 Average grade for subject {subject}: {average:.2f}\033[0m")
            else:
                print(f"\033[91m❌ No grades for subject {subject}.\033[0m") 

    def save_data(self, filepath):
        with open(filepath, 'w') as file:
            json.dump(self.subjects, file)
        print("\033[92m✅ Data successfully saved.\033[0m") 

    def load_data(self, filepath):
        try:
            with open(filepath, 'r') as file:
                self.subjects = json.load(file)
            print("\033[92m✅ Data successfully loaded.\033[0m")
        except FileNotFoundError:
            print("\033[91m❌ File not found. No data loaded.\033[0m")

def main_menu():
    filepath = "gradebook.json"
    gradebook = GradeBook()
    gradebook.load_data(filepath)

    while True:
        print("\nMain Menu:")
        print("1. Add grade 🗳️")
        print("2. Add subject 📦")
        print("3. Delete subject 🗑️")
        print("4. Delete grade 🗑️")
        print("5. Show average grades 📋")
        print("6. Save data 🛟")
        print("7. Exit 🚪")

        choice = input("Please choose an option (1-7): ")

        if choice == "1":
            subject = input("Please enter the subject: ")
            grade = input("Please enter the grade: ")

            try:
                grade = float(grade)
                gradebook.add_grade(subject, grade)
            except ValueError:
                print("\033[91m❌ Invalid input. Please enter a valid number for the grade.\033[0m")  
        elif choice == "2":
            subject = input("Please enter the subject: ")
            gradebook.add_subject(subject)
        elif choice == "3":
            subject = input("Please enter the subject to delete: ")
            gradebook.delete_subject(subject)
        elif choice == "4":
            subject = input("Please enter the subject: ")
            grade = input("Please enter the grade to delete: ")

            try:
                grade = float(grade)
                gradebook.delete_grade(subject, grade)
            except ValueError:
                print("\033[91m❌ Invalid input. Please enter a valid number for the grade.\033[0m") 
        elif choice == "5":
            gradebook.show_averages()
        elif choice == "6":
            gradebook.save_data(filepath)
        elif choice == "7":
            print("\033[92m✅ Exiting the program.\033[0m")  
            break
        else:
            print("\033[91m❌ Invalid choice. Please enter a number between 1 and 7.\033[0m")  

if __name__ == "__main__":
    main_menu()
