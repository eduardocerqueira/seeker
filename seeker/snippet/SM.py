#date: 2025-05-21T17:12:21Z
#url: https://api.github.com/gists/bb4743815701e2ce930c06ca98f46c6b
#owner: https://api.github.com/users/clydestories

from datetime import *
import os

shifts = []
prohibited_symbol = ["!", "@", "№", "$", "%", "^", "&", "*", "=", "+", "<", ">", "?", "#"]

save_file = open("save.txt", "r", encoding="utf-8")

while True:
    shift = dict()
    while True:
        line = save_file.readline().split("|")
        if len(line) < 2:
            break
        if line[1][-1] == "\n":
            line[1] = line[1][0:-1]
        if line[0] == "shift_start" or line[0] == "shift_end":
            line[1] = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
        elif line[0] == "hourly_rate" or line[0] == "profit_sum":
            line[1] = int(line[1])
        shift[line[0]] = line[1]
    if shift:
        shifts.append(shift)
    if save_file.tell() == os.stat("save.txt").st_size:
        break


def save_data():
    file = open("save.txt", "w", encoding="utf-8")
    for shift in shifts:
        for pair in shift:
            file.write(f"{pair}|{shift[pair]}\n")
        file.write("\n")


def add_shift():
    name = input("Введите ФИО: ")
    for letter in name:
        for symbol in prohibited_symbol:
            if letter == symbol:
                print("ОШИБКА: Найден запрещенный символ, операция прервана.")
                return
    try:
        shift_start_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        shift_start = datetime.strptime(shift_start_str, "%d/%m/%y %H:%M")
        shift_end_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        shift_end = datetime.strptime(shift_end_str, "%d/%m/%y %H:%M")
    except ValueError:
        print("ОШИБКА: Некорректный формат даты и времени, операция прервана.")
        return

    shift = {
        "full_name": name,
        "shift_start": shift_start,
        "shift_end": shift_end,
        "hourly_rate": int(input("Часовая ставка: ")),
        "profit_sum": int(input("Сумма продаж: "))
    }
    shifts.append(shift)
    save_data()


def show_all_shifts():
    if len(shifts) == 0:
        print("\nЗаписанных смен не найдено\n")
        return
    for shift in shifts:
        show_shift(shift)


def show_shift(shift):
    print(f"\nФИО Сотрудника - {shift['full_name']}")
    print(f"Начало смены - {shift['shift_start']}")
    print(f"Конец смены - {shift['shift_end']}")
    print(f"Начало смены - {shift['hourly_rate']}")
    print(f"Начало смены - {shift['profit_sum']}\n")


def delete_shift():
    if len(shifts) == 0:
        print("\nЗаписанных смен не найдено\n")
        return
    show_all_shifts()
    name = input("Введите ФИО сотрудника, чью смену надо удалить: ")
    shifts_by_name = find_shifts_by_name(name)
    if len(shifts_by_name) == 0:
        return
    for i in range(len(shifts_by_name)):
        print(f"Смена номер {i}\n")
        show_shift(shifts_by_name[i])
    try:
        shift_index_to_delete = int(input("Введите номер смены для удаления из перечня выше: "))
        shifts.remove(shifts_by_name[shift_index_to_delete])
    except ValueError:
        print("ОШИБКА: Необходимо ввести число, операция прервана")
        return
    except IndexError:
        print("ОШИБКА: Такого номера смены нет в перечне, операция прервана")
        return
    print("Смена успешно удалена")
    save_data()


def caculate_work_hours():
    result = 0
    show_all_shifts()
    name = input("Введите ФИО сотрудника, чью смену надо удалить: ")
    shifts_by_name = find_shifts_by_name(name)
    try:
        date_start_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_start = datetime.strptime(date_start_str, "%d/%m/%y %H:%M")
        date_end_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_end = datetime.strptime(date_end_str, "%d/%m/%y %H:%M")
    except ValueError:
        print("ОШИБКА: Некорректный формат даты и времени, операция прервана.")
        return
    target_shifts = get_shifts_between_dates(shifts_by_name, date_start, date_end)
    for shift in target_shifts:
        result += (shift["shift_end"] - shift["shift_start"]).total_seconds() / 3600
    print(
        f"По вашему запросу между {date_start} и {date_end} сотрудником {name} было отработано {int(round(result, 0))} часов")


def caculate_bonus():
    sells_sum = 0
    name = input("Введите ФИО сотрудника, чью смену надо удалить: ")
    shifts_by_name = find_shifts_by_name(name)
    try:
        date_start_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_start = datetime.strptime(date_start_str, "%d/%m/%y %H:%M")
        date_end_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_end = datetime.strptime(date_end_str, "%d/%m/%y %H:%M")
    except ValueError:
        print("ОШИБКА: Некорректный формат даты и времени, операция прервана.")
        return
    target_shifts = get_shifts_between_dates(shifts_by_name, date_start, date_end)
    for shift in target_shifts:
        sells_sum += shift["profit_sum"]
    print(f"По вашему запросу между {date_start} и {date_end} "
          f"сотрудник {name} должен получить премию в размере {int(round(sells_sum * 0.03, 0))} рублей")


def caculate_salary():
    salary = 0
    name = input("Введите ФИО сотрудника, чью смену надо удалить: ")
    shifts_by_name = find_shifts_by_name(name)
    try:
        date_start_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_start = datetime.strptime(date_start_str, "%d/%m/%y %H:%M")
        date_end_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_end = datetime.strptime(date_end_str, "%d/%m/%y %H:%M")
    except ValueError:
        print("ОШИБКА: Некорректный формат даты и времени, операция прервана.")
        return
    target_shifts = get_shifts_between_dates(shifts_by_name, date_start, date_end)
    for shift in target_shifts:
        hours = (shift["shift_end"] - shift["shift_start"]).total_seconds() / 3600
        salary += hours * shift["hourly_rate"]
    print(f"По вашему запросу между {date_start} и {date_end} "
          f"сотрудник {name} заработал на ставке {int(round(salary, 0))} рублей")


def caculate_restaurant_profit():
    sum = 0
    try:
        date_start_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_start = datetime.strptime(date_start_str, "%d/%m/%y %H:%M")
        date_end_str = input("Дата и время начала смены в формате ДД/ММ/ГГ ЧЧ:ММ")
        date_end = datetime.strptime(date_end_str, "%d/%m/%y %H:%M")
    except ValueError:
        print("ОШИБКА: Некорректный формат даты и времени, операция прервана.")
        return
    target_shifts = get_shifts_between_dates(shifts, date_start, date_end)
    if not target_shifts:
        return
    for shift in target_shifts:
        sum += shift["profit_sum"]
    print(f"Смены в ресторане с {date_start} по {date_end} принесли {sum} рублей")


def find_shifts_by_name(name):
    shifts_by_name = []
    for shift in shifts:
        if shift["full_name"] == name:
            shifts_by_name.append(shift)
    if len(shifts_by_name) == 0:
        print("Смен по этому имени не было найдено")
        return []
    return shifts_by_name


def get_shifts_between_dates(target_shifts, date_start, date_end):
    result = []
    for shift in target_shifts:
        if date_start <= shift["shift_start"] <= date_end and date_start <= shift["shift_end"] <= date_end:
            result.append(shift)
    if result:
        return result
    else:
        print("Смен в этом диапазоне дат не было найдено")
        return []


while True:
    print("Список команд: ")
    print("1 - Добавить смену")
    print("2 - Удалить смену")
    print("3 - Рассчитать количество отработанных часов")
    print("4 - Рассчитать зарплату сотрудника без учёта бонуса")
    print("5 - Рассчитать надбавку от личных продаж")
    print("6 - Рассчитать выручку всего заведения")
    print("7 - Вывести все смены: ")
    print("0 - Выход из программы")
    command = input("Ваша команда: ")
    print()
    if command == "1":
        add_shift()
    elif command == "2":
        delete_shift()
    elif command == "3":
        caculate_work_hours()
    elif command == "4":
        caculate_salary()
    elif command == "5":
        caculate_bonus()
    elif command == "6":
        caculate_restaurant_profit()
    elif command == "7":
        show_all_shifts()
    elif command == "0":
        save_data()
        print("До новых встреч!")
        break
    else:
        print("Введена неверная команда, попробуйте еще раз")

    input("\nНажмите enter для продолжения...")
    print()
