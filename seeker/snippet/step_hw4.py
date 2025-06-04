#date: 2025-06-04T17:10:48Z
#url: https://api.github.com/gists/37833f42220ce6db0ac46045f14479a8
#owner: https://api.github.com/users/dimtha

print("\n\033[31mОбчислення\033[0m \n")

num1 = int(input("Введіть перше число: "))
num2 = int(input("Введіть друге число: "))

print(f"+ : {num1 + num2} \n* : {num1 * num2} \n% : {num2 / 100 * num1} \nS : {(1/2) * num1 * num2}")

print("\n\033[31mДохід та витрати\033[0m \n")

zp = int(input("Скільки Ви заробляєте?: "))
credit = int(input("Скіки Ви сплачуєте по своїх необхідних кредитах?: "))
rent = int(input("Скільки Ви сплачуєте аренди за особняк?: "))

rest = print(f'Решта коштів після всіх виплат складає: {zp - credit - rent}')

print("\n\033[31mРозрахунок вартості поїздки\033[0m \n")

distance = int(input("Введіть кількість КМ від місця посадки до кінцевої зупинки: "))
consumption = int(input("Введітькількість витраченого палива на 100 км: "))
fuel_cost = int(input("Введіть вартість палива у Вашому регіоні: "))

print(f'Вартість поїздки складає: {((distance * consumption) / 100) * fuel_cost}')