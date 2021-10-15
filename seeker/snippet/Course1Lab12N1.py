#date: 2021-10-15T17:10:34Z
#url: https://api.github.com/gists/a832d197042c0c7a3c0132a7bc09b2ed
#owner: https://api.github.com/users/Menelaii

class Cathedra:
    def __init__(self, id, name, teachers_count, disciplines_id):
        self.id = id
        self.name = name
        self.teachers_count = teachers_count
        self.disciplines_id = disciplines_id


class Discipline:
    def __init__(self, id, name, lecture_count, practice_count, coursework, form_of_control):
        self.id = id
        self.name = name
        self.lecture_count = lecture_count
        self.practice_count = practice_count
        self.coursework = coursework
        self.form_of_control = form_of_control


class DataBase:
    def __init__(self):
        self.cathedras = []
        self.disciplines = []

    def add_discipline(self):
        print('Введите: название, кол-во лекций, кол-во практик, курсовая работа?(+ / -), форма контроля')
        id = len(self.disciplines)
        name, lecture_count, practice_count, coursework, form_of_control = input().split(',')
        self.disciplines.append(Discipline(id, name, lecture_count, practice_count, coursework, form_of_control))

    def add_cathedra(self):
        if(len(self.disciplines) > 0):
            print('Введите: название, кол-во преподавателей')
            id = len(self.cathedras)
            name, teachers_count  = input().split(',')
            for i in self.disciplines:
                print(f'{i.id}) {i.name}')
            print('Введите: id дисциплин через запятую')
            id_list = input().split(',')
            for i in range(len(id_list)):
                id_list[i] = int(id_list[i])
            self.cathedras.append(Cathedra(id, name, teachers_count, id_list))
        else:
            print('добавьте сначала дисциплины')

    def show_disciplines(self):
        for i in self.disciplines:
            if i.form_of_control == 'экзамен' and i.coursework == '+':
                print(i.name)


data_base = DataBase()
user_input = -1
while user_input != 0:
    print('0 - Выйти\n1 - Добавить кафедру\n2 - добавить дисциплину\n3 - Вывести список дисциплин')
    user_input = int(input())
    if user_input == 1:
        data_base.add_cathedra()
    elif user_input == 2:
        data_base.add_discipline()
    elif user_input == 3:
        data_base.show_disciplines()