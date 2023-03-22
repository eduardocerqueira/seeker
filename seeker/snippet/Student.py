#date: 2023-03-22T16:53:29Z
#url: https://api.github.com/gists/c69278775f19315dd1310216b222458e
#owner: https://api.github.com/users/ashtanyuk

import datetime

class Student():
  def __init__(self, name, dob, number):
    self.name = name
    self.birth_date = dob
    self.phone_number = number

  def age_calculator(self):
    current_date = datetime.datetime.now().date()
    student_birthdate = datetime.datetime.strptime(self.birth_date, "%m/%d/%y").date()

    delta = current_date - student_birthdate
    age = int(delta.days/365)
    return age