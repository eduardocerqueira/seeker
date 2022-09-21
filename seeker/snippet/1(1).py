#date: 2022-09-21T17:04:01Z
#url: https://api.github.com/gists/26ac60f7c4a1d320d72c57dd295065d3
#owner: https://api.github.com/users/MiDemGG

УВВ-111
Правктика 1

Прямоугольник задается координатами двух его противоположных углов
# (x1, y1), (x2, y2). Рассчитайте его площадь и периметр.
# Ввод считается из консоли. Числа x1, y1, x2 и y2 задаются по одному за раз.
# Выходные данные отображаются на консоли и должны содержать две
# строки с площадью и периметром. Выводиться должно 2 знака после
# точки.

import math

x1 = int(input())
y1 = int(input())
x3 = int(input())
y3 = int(input())

y2 = y1
x2 = x3
y4 = y3
x4 = x1

side1 = math.fabs(x1 - x2)
side2 = math.fabs(y1 - y4)

print('площадь', str(side1 * side2) + '0')
print('периметр', str((side1 + side2)*2) + '0')