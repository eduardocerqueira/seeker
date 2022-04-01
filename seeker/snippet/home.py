#date: 2022-04-01T17:12:33Z
#url: https://api.github.com/gists/de4cb69e6bf19d96e30979773b30f8fb
#owner: https://api.github.com/users/Pechenkaaaaaaaaaa

class Humen:
    def __init__(self, name, last, city):
        self.name = name
        self.last = last
        self.city = city
        self.email = f'{name}.{last}@company.ua'


    def fullname(self):
        return f'{self.name} {self.last}'

hu_1 = Humen('Nazar', 'Petrov', 'Kiev')

class Adult(Humen):

    def __init__(self, name, last, city, favoritecafe):
        super().__init__(name, last, city)
        self.favoritecafe = favoritecafe

adu_1 = Adult('Artem', 'Pechkin', 'Lviv', 'LvivCroasan')
adu_2 = Adult('Masha', 'Shevchenko', 'Kiev', 'GoodCafe')

class Student(Adult):

    def __init__(self, name, last, city, favoritecafe, university):
        super().__init__(name, last, city, favoritecafe)
        self.university = university

stu_1 = Student('Dasha', 'Melnichenko', 'Kiev', 'Cafe', 'Pty')

class Child(Humen):

    def __init__(self, name, last, city,  favoritetoy):
        super().__init__(name, last, city, )
        self.favoritetoy = favoritetoy

chi_1 = Child('Macsim', 'Maksimov', 'Kiev',  'Car')

class Worker(Adult):

    def __init__(self, name, last, city, favoritecafe, cualification ):
        super().__init__(name, last, city, favoritecafe)
        self.cualification = cualification

wor_1 = Worker('Larisa', 'Last', 'Kiev', 'geyartem','teacher')

class Teacher(Worker):

    def __init__(self, name, last, city, favoritecafe, cualification):
        super().__init__(name, last, city, favoritecafe, cualification)


te_1 = Teacher('Larisa', 'Last', 'Kiev', 'geyartem', 'teacher')


class TeacherWork(Teacher):

    def __init__(self, name, last, city, favoritecafe, cualification, auditoria):
        super().__init__(name, last, city, favoritecafe, cualification)
        self.auditoria = auditoria

tewor_1 = TeacherWork('Larisa', 'Last', 'Kiev', 'geyartem', 'teacher' , '100 students')


print(isinstance(hu_1, Humen))
print("_______________")
print(isinstance(adu_1, Adult))
print(isinstance(adu_2, Adult))
print("_______________")
print(isinstance(stu_1, Student))
print("_______________")
print(isinstance(chi_1, Child))
print("_______________")
print(isinstance(wor_1, Worker))
print("_______________")
print((te_1, Teacher))
print("_______________")
print(isinstance(tewor_1, TeacherWork))
