#date: 2025-03-07T16:43:49Z
#url: https://api.github.com/gists/2330a1648c388f209e087b7754f04b4b
#owner: https://api.github.com/users/CandyCatUWU

import datetime
import os

tasks = []
save_file = open("save.txt", "r")
while True:
    task = dict()
    while True:
        line = save_file.readline().split(":")
        if len(line) < 2:
            break
        if line[1][-1] == "\n":
            line[1] = line[1][0:-1]
        task[line[0]] = line[1]
    tasks.append(task)
    if save_file.tell() == os.stat("save.txt").st_size:
        break


def save_data():
    file = open("save.txt", "w")
    for task in tasks:
        for pair in task:
            file.write(f"{pair}:{task[pair]}\n")
        file.write("\n")


def add_task():
    while True:
        due_date = input("Дата исполнения (В формате дд.мм.гггг):").split(".")
        for i in due_date:
            if i.isnumeric() == False:
                print("Некорректный формат ввода даты")
                return
        due_date = [int(date) for date in due_date]
        due_date.reverse()
        due_date = datetime.date(due_date[0], due_date[1], due_date[2])
        if due_date > datetime.date.today():
            break
        else:
            print("Дата исполнения должна быть в будущем")

    task = {
        "name": input("Название задачи: "),
        "username": input("Ваше имя пользователя: "),
        "description": input("Описание задачи: "),
        "status": input("Статус задачи: "),
        "notes": [],
        "due_date": due_date
    }
    tasks.append(task)
def find_task():
    show_tasks(tasks)
    key=""
    find_key= input("Выберете характеристику поиска:\n"
                    "1 - Название задачи\n"
                    "2 - Имя пользователя\n"
                    "3 - Описание задачи\n"
                    "4 - Статус задачи\n"
                    "5 - Дата\n")
    if find_key=="1":
        key="name"
    elif find_key=="2":
        key="username"
    elif find_key=="3":
        key="description"
    elif find_key=="4":
        key="status"
    elif find_key=="5":
        key="due_date"
    else:
        print("Некорректный ввод")
        return
    find_tasks=[]
    find_value=input("Введите значение для поиска: ")
    for task in tasks:
        if key=="due_date":
            find_tasks.append(f"{datetime.datetime(task[key]).day}."
                              f"{datetime.datetime(task[key]).month}."
                              f"{datetime.datetime(task[key]).year}")
        if task[key]==find_value:
            find_tasks.append(task)
    show_tasks(find_tasks)

def update_task():
    show_tasks(tasks)
    task_name=input("Введите название задачи: ")
    key=""
    task_to_update=None
    for task in tasks:
        if task["name"]==task_name:
            task_to_update=task
    if task_to_update==None:
        print("Такой задачи нет. Попробуйте ещё раз.")
        return
    key_to_update=input("Введите характеристику для обновления: "
                         "1 - Название задачи\n"
                        "2 - Имя пользователя\n"
                        "3 - Описание задачи\n"
                        "4 - Статус задачи\n"
                        "5 - Дата\n")
    if key_to_update=="1":
        key="name"
    elif key_to_update=="2":
        key="username"
    elif key_to_update=="3":
        key="description"
    elif key_to_update=="4":
        key="status"
    elif key_to_update=="5":
        key="due_date"
    else:
        print("Некорректный ввод")
        return
    new_value=input ("Введите новое значение характеристики (При введении даты использовать формат: дд.мм.гггг): ")
    if key == "due_date":
        new_value=new_value.split(".")
        if len(new_value)!=3:
            print("Некорректный формат ввода даты")
            return
        for i in new_value:
            if i.isnumeric()==False:
                print("Некорректный формат ввода даты")
                return
        new_value = datetime.date(int(new_value[0]), int(new_value[1]), int(new_value[2]))
    task_to_update[key]=new_value
    print("Успешно обновлено!")
def show_tasks(task_list):
    if len(task_list) == 0:
        print("\nУ вас нет задач\n")
        return
    for task in task_list:
        print(f"Название задачи: {task['name']}")
        print(f"Имя пользователя: {task['username']}")
        print(f"Описание задачи: {task['description']}")
        print(f"Статус задачи: {task['status']}")
        print(f"Дата исполнения: {task['due_date']}\n")


def delete_task():
    if len(tasks)==0:
        print("\nУ вас нет задач\n")
        return
    show_tasks(tasks)
    name_to_delete = input("Название задачи для удаления: ")
    is_found = False
    for task in tasks:
        if task["name"] == name_to_delete:
            tasks.remove(task)
            print("Удалено успешно!")
            is_found = True
            break
    if is_found==False:
        print("Задача по этому имени не найдена")

while True:
    print("Что хотите сделать?\n"
          "1 - Добавить задачу\n"
          "2 - Удалить задачу\n"
          "3 - Просмотреть все задачи\n"
          "4 - Найти задачу\n"
          "5 - Обновить задачу\n"
          "0 - Выйти")
    user_input = input("Введите номер команды: ")
    if user_input == "1":
        add_task()
    elif user_input == "2":
        delete_task()
    elif user_input == "3":
        show_tasks(tasks)
    elif user_input=="4":
        find_task()
    elif user_input=="5":
        update_task()
    elif user_input == "0":
        print("Досвидания!")
        save_data()
        break
    else:
        print("Такой команды нет. Попробуйте ещё раз.")
        continue
