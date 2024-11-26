#date: 2024-11-26T17:00:04Z
#url: https://api.github.com/gists/495842add0b5554d6b588ac3b824b913
#owner: https://api.github.com/users/MrNtex

import json

with open('Pizzeria\\menu.json', 'r') as file:
    menu = json.load(file)

list_of_pizzas = menu['menu']

def display_menu():
    for pizza in list_of_pizzas:
        print(f"Pizza {pizza['pizza']}")
        print(f"Składniki {', '.join(pizza['dodatki'])}")
        print(f"Ceny: Mała:{pizza['ceny']['S']}zł Średnia:{pizza['ceny']['M']}zł Duża:{pizza['ceny']['L']}zł")
    input("Wciśnij enter aby wrócić do ekranu głównego")

order = []

def add_to_order():
    print("Którą pizzę chcesz zamówić (podaj numer): ")
    for i,pizza in enumerate(menu['menu']):
        print(f"{i+1}. {pizza['pizza']}")
    pizza_number = int(input())

    print("Podaj rozmiar pizzy (S/M/L): ")
    pizza_size = input()

    print("Podaj ilość pizz: ")
    pizza_amount = int(input())

    order.append({'pizza': menu['menu'][pizza_number-1]['pizza'], 'size': pizza_size, 'amount': pizza_amount})

def calculate_cost():
    cost = 0
    for ordered_pizza in order:
        for pizza in menu['menu']:
            if pizza['pizza'] == ordered_pizza['pizza']:
                cost += ordered_pizza['amount'] * pizza['ceny'][ordered_pizza['size']]

    return cost
def send_order():
    print("Twoje zamówienie:")
    for ordered_pizza in order:
        print(f"{ordered_pizza['amount']} x {ordered_pizza['pizza']} [{ordered_pizza['size']}] = {calculate_cost(ordered_pizza)}zł")
    print(f"Razem: {calculate_cost(ordered_pizza)}zł")
    order.clear()

def main_page():
    print("Witaj na stronie pizzeri u Vita")
    print("Wybierz co chcesz zrobić")
    print("1. Wyświetl menu")
    print("2. Dodaj pizze do zamówienia")
    print("3. Wyczyść zamówienie")
    print("4. Wyślij zamówienie")
    print("5. Zakończ")
    option = input("")
    if option == '1':
        display_menu()
    elif option == '2':
        add_to_order()
    elif option == '3':
        order.clear()
    elif option == '4':
        pass
    elif option == '5':
        pass
    else:
        print("Podano złą opcję")
    main_page()

main_page()

  