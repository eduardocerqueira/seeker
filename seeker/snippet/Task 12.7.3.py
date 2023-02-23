#date: 2023-02-23T17:01:00Z
#url: https://api.github.com/gists/bf62c7a26d055c4120089d5662c5f7c2
#owner: https://api.github.com/users/sssoff

per_cent = {'ТКБ': 5.6, 'СКБ': 5.9, 'ВТБ': 4.28, 'СБЕР': 4.0}


deposit = []
money = int(input('Сумму которую человек планирует положить'))
for values in per_cent:
    deposit.append(str(per_cent[values] * money/100))
print(deposit)

deposit_i = max(deposit)
print('Максимальная сумма, которую вы можете заработать - ', deposit_i)
