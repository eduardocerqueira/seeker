#date: 2023-01-31T16:49:55Z
#url: https://api.github.com/gists/43725813cb6bffae2360038f5896dcd4
#owner: https://api.github.com/users/abrkriss

# Exercise_1
a = input('Введите Ваше имя ')
b = input('Введите Ваш пароль для входа в систему ')
c = input('Введите Ваш возраст ')
print(f"Ваше имя: {a} Ваш пароль {b} Ваш возраст {c}")
print()
# Exercise_2
a = int(input('Введите время в секундах '))
print(f'Время в формате ч:м:с {a / 3600} : {a / 60} : {a}')
print()
# Exercise_3
n = int(input('Введите целое положительное число '))
print(f'{n} + {n + n * 10} + {n * 100 + n * 10 + n} = {n * 123}')
print()
# Exercise_4
a = float(input('Введите сумму выручки '))
b = float(input('Введите сумму издержек '))
c = a - b
if c >= b:
    print(f'Прибыль - выручка больше издержек и равна {c}')
    print(f'Рентабельность {c / a}')
else:
    print(f'Убытки - издержки больше выручки')

if c >= b:
    d = int(input('Введите количество сутрудников '))
    print(f'Прибыль в расчете на одного сотрудника {c / d}')
