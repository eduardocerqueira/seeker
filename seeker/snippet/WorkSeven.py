#date: 2024-11-21T17:11:45Z
#url: https://api.github.com/gists/910753385983c7fc24562855f2af3145
#owner: https://api.github.com/users/EXODEZ

# 6 задание
number = input("Введите десятизначное число: ")

if len(number) != 10 or not number.isdigit():
    print("Ошибка: Введите корректное десятизначное число.")
else:
    digits = [int(digit) for digit in number]

    max_digit = max(digits)
    min_digit = min(digits)

    print(f"Самая большая цифра: {max_digit}")
    print(f"Самая маленькая цифра: {min_digit}")

# 7 задание
ticket_number = input("Введите шестизначное число: ")

if len(ticket_number) != 6 or not ticket_number.isdigit():
    print("Ошибка: Введите корректное шестизначное число.")
else:
    first_half = ticket_number[:3]
    second_half = ticket_number[3:]
    sum_first_half = sum(int(digit) for digit in first_half)
    sum_second_half = sum(int(digit) for digit in second_half)

    if sum_first_half == sum_second_half:
        print("Счастливый билет")
    else:
        print("Несчастливый билет")

