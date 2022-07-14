#date: 2022-07-14T17:20:49Z
#url: https://api.github.com/gists/238613edd0fb5033ab12fafa2275fd79
#owner: https://api.github.com/users/Wilidon

import random

doctor = dict.fromkeys(['Специальность', 'Опыт работы', 'Пациенты'])

doctor['Фамилия'] = 'Иванов'
doctor['Имя'] = 'Петр'
doctor['Отчество'] = 'Михайлович'
print(doctor)

doctor['Специальность'] = 'Стоматолог'
doctor['Опыт работы'] = '5 лет'
doctor['Пациенты'] = ['Иванова', 'Козлова']
# Вывод ключей словаря
print(doctor.keys())
# Вывод значений словаря
print(doctor.values())
# deleting Отчество
del doctor['Отчество']
# вывод на экран значений по ключу фамилия и специальность
print(f"Вывод значений по ключу фамилия и специальность"
      f" {doctor['Фамилия'], doctor['Специальность']}")

# task 2
def make_doctor_01(last_name, name, middle_name):
    doctor = {'Фамилия': last_name, 'Имя': name, 'Отчество': middle_name}
    return doctor


first_doc = make_doctor_01('Куликова', 'Мария', 'Иванова')
first_doc['Пациенты'] = []
print(first_doc)

# task 3
def make_doctor_02(last_name, name, middle_name):
    doctor = {'Фамилия': last_name, 'Имя': name, 'Отчество': middle_name,
              'Пациенты': []}
    return doctor


second_doc = make_doctor_02('Османов', 'Борис', 'Витальевич')
print(second_doc)

# task 4
doctors = [doctor, first_doc, second_doc]
print(doctors)


# 5 task
def new_patient(city, name='Неизвестный', last_name=''):
    patient = {'Город': city, 'Имя': name, 'Фамилия': last_name}
    return patient


first_patient = new_patient('Москва', 'Михиал', 'Великий')
print(first_patient)

# task 6
def add_age(patient,year):
    patient['Год рождения'] = year
    return patient


first_patient = add_age(first_patient, 2001)
print(first_patient)

for i in range (3):
    doctors[i] = add_age(doctors[i], 1960 + random.randint(0,30))
print(doctors)

for i in range(3):
    print(2022 - doctors[i].get('Год рождения'))

print(f"Вывод возраста в строку {doctors[0].get('Год рождения')} , "
      f"{doctors[1].get('Год рождения')}, {doctors[1].get('Год рождения')}))