#date: 2023-08-03T16:55:11Z
#url: https://api.github.com/gists/fb20f376a676a4776efbabe67e42258e
#owner: https://api.github.com/users/greatvijay

#Challenge 3
class Student:
    def __init__(self):
        self._name = None
        self._rollNumber = None

    def setName(self, name):
        self._name = name

    def getName(self):
        return self._name

    def setRollNumber(self, rollNumber):
        self._rollNumber = rollNumber

    def getRollNumber(self):
        return self._rollNumber

# Testing the Student class
student = Student()
student.setName("John Doe")
student.setRollNumber("12345")

print(student.getName())       # Output: John Doe
print(student.getRollNumber()) # Output: 12345
