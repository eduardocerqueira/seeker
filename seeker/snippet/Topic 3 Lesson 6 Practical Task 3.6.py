#date: 2023-09-12T16:51:17Z
#url: https://api.github.com/gists/ac4e919bf9f627d4508005605d86e4fe
#owner: https://api.github.com/users/ITagada

import collections


def lesson6_task1():
    n = int(input("Введите число: "))
    def factorial(n):
        if n == 1:
            return 1
        return n * factorial(n - 1)
    facts_list = []
    for i in range(factorial(n), 0, -1):
        facts_list.append(factorial(i))
    print(facts_list)

# lesson6_task1()

def lesson6_task2():
    pets_db = {1:
                   {"Мухтар":
                        {"Вид питомца": "Собака",
                         "Возраст питомца": 9,
                         "Владелец питомца": "Павел"},
                    },
               2:
                   {"Каа":
                        {"Вид питомца": "желторотыйпитон",
                         "Возраст питомца": 19,
                         "Владелец питомца": "Саша"},
                    },
               }
    command = str

    def create(pets_db: dict):
        last = collections.deque(pets_db, maxlen=1)[0]
        pet_name = input('Введите имя питомца: ')
        pet_type = input('Введите вид питомца: ')
        pet_age = int(input('Введите возраст питомца: '))
        owner = input('Введите имя владельца: ')
        last += 1
        pets_db[last] = {pet_name: {'Вид питомца': pet_type,
                                   'Возраст питомца': pet_age,
                                   'Владелец питомца': owner}
                         }

    def get_pet_id(pets_db: dict, id):
        if id in pets_db.keys():
            for i in pets_db[id].keys():
                k = pets_db[id]
                age = k[i]['Возраст питомца']
                def correct_age(age):
                    return {
                        age < 0: '',
                        age % 10 == 0: 'лет',
                        age % 10 == 1: 'год',
                        age % 10 > 1 and age % 10 < 5: 'года',
                        age % 10 > 4: 'лет',
                        age % 100 > 10 and age % 100 < 20: 'лет'
                    }[True]
                for j in k:
                    return str(f'Это {k[i]["Вид питомца"]} по кличке {j}. '
                               f'Возраст питомца: {k[i]["Возраст питомца"]} {correct_age(age)}. '
                               f'Имя владельца: {k[i]["Владелец питомца"]}')
        return False

    def update(pest_db: dict, id):
        if id in pest_db.keys():
            change = input('Введите параметр, который хотите изменить: ')
            if change == 'Имя питомца':
                new_name = input('Введите новое имя питомца: ')
                pest_db[id] = new_name
            elif change == 'Вид питомца':
                new_type = input('Введите новый вид питомца: ')
                for i in pest_db[id].keys():
                    k = pest_db[id]
                    k[i]['Вид питомца'] = new_type
            elif change == 'Возраст питомца':
                new_age = int(input('Введите новый возраст питомца: '))
                for i in pest_db[id].keys():
                    k = pest_db[id]
                    k[i]['Возраст питомца'] = new_age
            elif change == 'Владелец питомца':
                new_owners_name = input('Введите новое имя владельца питомца: ')
                for i in pest_db[id].keys():
                    k = pest_db[id]
                    k[i]['Владелец питомца'] = new_owners_name
            else:
                return str('Ошибка')

    def delete(pets_db: dict, id):
        del pets_db[id]
        return str(f'Запись {id} удалена!')

    while command != 'stop':
        command = input('Введите команду: ')
        if command == 'create':
            create(pets_db)
        elif command == 'read':
            id = int(input('Введите номер питомца: '))
            print(get_pet_id(pets_db, id))
        elif command == 'update':
            id = int(input('Введите номер питомца, которого хотите изменить: '))
            update(pets_db, id)
        elif command == 'delete':
            id = int(input('Введите номер питомцаб которого хотите удалить: '))
            print(delete(pets_db, id))
        elif command == 'stop':
            print('Спасибо, что воспользовались нашим сервисом')
            break
        else:
            print('Ошибка!!!')
    print(pets_db)

# lesson6_task2()