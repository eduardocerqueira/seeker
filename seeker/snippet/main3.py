#date: 2022-12-08T16:50:06Z
#url: https://api.github.com/gists/b5af6fb3f602e029d6b83021dc2e3074
#owner: https://api.github.com/users/karchh


# Домашка №4
# Сдайте задание до: 8 дек., 20:00 +03:00 UTC

# 1. Задана натуральная степень k. Сформировать случайным образом список коэффициентов
# (значения от 0 до 100) многочлена и записать в файл многочлен степени k.
# Пример:
# если k = 2, то многочлены могут быть => 2*x² + 4*x + 5 = 0 или x² + 5 = 0 или 10*x² = 0


# from random import randint
# import itertools
#
# k = int(input('Задайте натуральную степень многочлена k: '))
#
# def get_ratios(k):
#     ratios = [randint(0, 100) for i in range(k + 1)]
#     while ratios[0] == 0:
#         ratios[0] = randint(1, 10)
#     return ratios
#
# def get_polynomial(k, ratios):
#     var = ['*x^'] * (k-1) + ['*x']
#     polynomial = [[a, b, c] for a, b, c in itertools.zip_longest(ratios, var, range(k, 1, -1), fillvalue='') if a != 0]
#     for x in polynomial:
#         x.append(' + ')
#     polynomial = list(itertools.chain(*polynomial))
#     polynomial[-1] = ' = 0'
#     return "".join(map(str, polynomial)).replace(' 1*x',' x')
#
# ratios = get_ratios(k)
# polynom = get_polynomial(k, ratios)
# print(polynom)



# 2. Даны два файла, в каждом из которых находится запись многочлена.
# Задача - сформировать файл, содержащий сумму многочленов.

from random import randint
import itertools

k = int(input('Задайте натуральную степень многочлена k: '))

def get_ratios(k):
    ratios = [randint(0, 100) for i in range(k + 1)]
    while ratios[0] == 0:
        ratios[0] = randint(1, 10)
    return ratios

def get_polynomial(k, ratios):
    var = ['*x^'] * (k-1) + ['*x']
    polynomial = [[a, b, c] for a, b, c in itertools.zip_longest(ratios, var, range(k, 1, -1), fillvalue='') if a != 0]
    for x in polynomial:
        x.append(' + ')
    polynomial = list(itertools.chain(*polynomial))
    polynomial[-1] = ' = 0'
    return "".join(map(str, polynomial)).replace(' 1*x',' x')

ratios = get_ratios(k)
polynom = get_polynomial(k, ratios)
print(polynom)

m = int(input('Задайте натуральную степень второго многочлена m: '))

second_polynom = get_polynomial(m, ratios)
print(second_polynom)

