#date: 2026-02-02T17:26:33Z
#url: https://api.github.com/gists/692b878e0e7a262a3a1bc34de6758035
#owner: https://api.github.com/users/basystaali-design

# 1
import datetime
# date_now = datetime.datetime.now()
# print(f"Поточна дата та час: {date_now}")

# 2
# date_str = input("Введіть дату: ")
# date_now = datetime.datetime.now()
# if date_str == date_now.strftime("%d.%m.%Y"):
#     date = datetime.strptime(date_str, "%d.%m.%Y")
#     print("Правильна дата")
# else:
#     print("Неправильна дата.")

# 3
# def age():
#     y = int(input("Введіть рік народження: "))
#     m = int(input("Введіть місяць народження: "))
#     d = int(input("Введіть день народження: "))
#     today = datetime.date.today()
#     birth_date = datetime.date(y, m, d)
#     age1 = today.year - birth_date.year
#     if (today.month, today.day) < (birth_date.month, birth_date.day):
#         age1 -= 1
#     return age1
# age1 = age()
# print("Вік користувача:", age)

# 4
# year = int(input("Рік: "))
# month = int(input("Місяць: "))
# day = int(input("День: "))
# year1 = int(input("Рік: "))
# month1 = int(input("Місяць: "))
# day1 = int(input("День: "))
# date = datetime.date(year, month, day)
# date1 = datetime.date(year1, month1, day1)
# d = (date - date1).days
# print("Різниця між датами:", {d}, "днів")