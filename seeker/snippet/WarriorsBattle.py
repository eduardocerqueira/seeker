#date: 2025-04-10T17:07:17Z
#url: https://api.github.com/gists/12f859956f1250e8d2e9c5c74b5768a1
#owner: https://api.github.com/users/MizAless

import random


class Warrior:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.armor = 50
        self.stamina = 300
        self.action = None  # 'attack' или 'defend'

    def choose_action(self):
        randomValue = random.randint(1, 2)

        if randomValue == 1:
            if self.stamina >= 10:
                self.action = 'attack'
            else:
                self.action = 'defend'
                print(f"{self.name} хотел атаковать, но у него нет сил!")
        else:
            self.action = 'defend'

    def perform_action(self, enemy):
        if self.action == 'attack' and enemy.action == 'attack':
            # Оба атакуют
            damage_self = random.randint(10, 30)
            damage_enemy = random.randint(10, 30)
            self.health -= damage_self
            enemy.health -= damage_enemy
            self.stamina -= 10
            enemy.stamina -= 10
            print(
                f"Оба атаковали! {self.name} потерял {damage_self} здоровья, {enemy.name} потерял {damage_enemy} здоровья.")

        elif self.action == 'attack' and enemy.action == 'defend':
            # Этот атакует, враг защищается
            self.stamina -= 10

            if enemy.armor > 0:
                health_loss = random.randint(0, 20)
                armor_loss = random.randint(0, 10)
                enemy.health -= health_loss
                enemy.armor -= armor_loss
                print(
                    f"{self.name} атаковал, {enemy.name} защитился. Потеряно: {health_loss} здоровья и {armor_loss} брони.")
            else:
                health_loss = random.randint(10, 30)
                enemy.health -= health_loss
                print(
                    f"{self.name} атаковал, {enemy.name} защитился (броня кончилась). Потеряно: {health_loss} здоровья.")

        elif self.action == 'defend' and enemy.action == 'attack':
            # Этот защищается, враг атакует - логика обработается при вызове enemy.perform_action(self)
            pass

        else:  # Оба защищаются
            print(f"Оба защищаются - ничего не происходит.")


warrior1 = Warrior("Воин 1")
warrior2 = Warrior("Воин 2")

round_num = 1
while warrior1.health > 0 and  warrior2.health:
    print(f"\n=== Раунд {round_num} ===")

    # 1. Выбор действий
    warrior1.choose_action()
    warrior2.choose_action()

    print(f"{warrior1.name} выбрал: {warrior1.action}")
    print(f"{warrior2.name} выбрал: {warrior2.action}")

    # 2. Выполнение действий
    warrior1.perform_action(warrior2)
    warrior2.perform_action(warrior1)  # Обрабатываем случай
    round_num += 1

if warrior1.health > 0:
    print(warrior1.name + " победил!")
else:
    print(warrior2.name + " победил!")
