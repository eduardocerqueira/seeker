#date: 2023-08-25T17:04:02Z
#url: https://api.github.com/gists/f5b8eb9e515fb78aa2d1902ee2700fcf
#owner: https://api.github.com/users/MipoX

from peewee import *
from datetime import date

db = SqliteDatabase('simple.db')


class Person(Model):
    name_user = CharField(column_name='Name')
    birthday = DateField()

    class Meta:
        database = db


class Pet(Model):
    owner = ForeignKeyField(Person, backref='pets')
    name = CharField()
    type_pet = CharField()

    class Meta:
        database = db


db.connect()
db.create_tables([Person, Pet])

father_nicol = Person(name_user='Nikola', birthday=date(1990, 10, 15))
father_nicol.save()
mather_olga = Person.create(name_user="Olga", birthday=date(year=1965, month=12, day=22))
alex = Person.create(name_user='Alex', birthday=date(1995, 3, 19))

mather_olga.name_user = 'Olga A.'
mather_olga.save()

father_dog = Pet.create(owner=father_nicol, name='Sharik', type_pet='dog')
alex_cat = Pet.create(owner=alex, name='APysi', type_pet='cat')
alex_dog = Pet.create(owner=alex, name='Byblik', type_pet='dog')
alex_cat_2 = Pet.create(owner=alex, name='Fucker', type_pet='cat')

alex_cat_2.delete_instance()

alex_cat.owner = father_nicol
alex_cat.save()

mather_olga = Person.select().where(Person.name_user == 'Olga A.').get()
mather_olga_2 = Person.get(Person.name_user == 'Olga A.')

for person in Person.select():  # Перечислим всех людей в базе данных
    print(person.name_user)
print()

qwerty = Pet.select(Pet, Person).join(Person).where(Pet.type_pet == "dog")  # Перечислим всех dog и имя их владельца

for pet in qwerty:
    print(pet.name, pet.owner.name_user)
print()

for pet in Pet.select().join(Person).where(Person.name_user == 'Nikola'):
    print(pet.name)  # получим всех домашних животных, принадлежащих Nikola
print()

for pet in Pet.select().where(Pet.owner == father_nicol):
    print(pet.name)  # получим всех домашних животных, принадлежащих Nikola
print()

for pet in Pet.select().where(Pet.owner == father_nicol).order_by(Pet.name):
    print(pet.name)  # отсортированы в алфавитном порядке, добавив order_by() 
print()

for person in Person.select().order_by(Person.birthday.desc()): 
    # перечислим всех людей, от самого младшего до самого старшего
    print(person.name_user, person.birthday)
print('___')


d1970 = date(1970, 1, 1)
d1992 = date(1992, 1, 1)  # поиск тех | или иных значений
for person in Person.select().where((Person.birthday < d1970) | (Person.birthday > d1992)):
    print(person.name_user, person.birthday)


print('****')
for person in Person.select().where((Person.birthday.between(d1970, d1992))):  # поиск в диапазоне
    print(person.name_user, person.birthday)


print('####')
for person in Person.select():  # N+1 поиск и сортировка
    print(person.name_user, person.pets.count(), 'pets')
print('@@@@')


qwert = (Person.select(Person, fn.COUNT(Pet.id).alias("pet_count"))
         .join(Pet, JOIN.LEFT_OUTER).group_by(Person).order_by(Person.name_user))

for person in qwert:  # N поиск и сортировка
    print(person.name_user, person.pet_count, 'pets')
print()


qweqwe = Person.select().order_by(Person.name_user).prefetch(Pet)  # поиски и сортировка
for person in qweqwe:
    print(person.name_user)
    for pet in person.pets:
        print('    ', pet.name)
print()


expression = fn.Lower(fn.Substr(Person.name_user, 1, 1)) == 'n'  # поиску значений по символу
for person in Person.select().where(expression):
    print(person.name_user)
print()


db.close()
