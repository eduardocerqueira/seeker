#date: 2023-05-17T16:45:56Z
#url: https://api.github.com/gists/9e44057acfa24a10a382cce24dea48f1
#owner: https://api.github.com/users/voxelin

import random

above_zero_count = 0
temperatures = []

for i in range(7):
    temperatures.append(random.randint(-10, 5))

print("Температури:", temperatures)

for i in range(len(temperatures)):
    if temperatures[i] >= 0:
        above_zero_count += 1
        print("День:", i + 1, "-", "Температура:", temperatures[i])

print(above_zero_count)


# Знайти та вивести добуток найбільшого та найменшого елементів списку.
print("Завдання 2:", min(temperatures) * max(temperatures))

