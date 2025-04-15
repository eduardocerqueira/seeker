#date: 2025-04-15T17:02:00Z
#url: https://api.github.com/gists/355bebf87ae5d830da2e9c394cea8972
#owner: https://api.github.com/users/nikita-popov-java

### 2 задание
a = int(input('a = '))
b = int(input('b = '))
s = a + b
print('s = ', s)
### 3 задание
fio = input('Введите ФИО: ')
age = int(input('Введите возраст: '))
city = input('Введите ваш город: ')
Person = [fio, age, city]
print(Person)
newAge = age + 1
Person[1] = newAge
phone = int(input('Введите номер телефона: '))
Person.append(phone)
print(Person)
### 4 задание
v = int(input('Введите начальную скорость тела: '))
t = int(input('Введите время: '))
g = 9.8
y = v * t - ((g * t**2) / 2)
print("y={:.2f}".format(y))
### 5 задание
from math import sqrt
d = 8
e = 2
print(sqrt(d ** e))
### 6 задание
import time
from math import pi, atanh, sqrt
t1 = time.perf_counter()
a = 6378137
c = 6356752.314245
e = sqrt(1 - (c**2 / a**2))
S1 = (2 * pi * a**2) * (1 + ((1 - e**2) / e)*atanh(e))
R = 6371000
S2 = 4 * pi * R**2
t2 = time.perf_counter()
print('S1 = ', S1)
print('S2 = ', S2)
print('Разность между площадами: ', S1 - S2)
print('В процентах: ', (S1 - S2) / S2 * 100)
print('Время, потребовавшееся на вычисление: ', t2 - t1)
### 9 задание
##########
# a
##########
L = [1, 1]
n = int(input('n = '))

for x in range (n - 2):
    a = L[x] + L[x + 1]
    L.append(a)
        
print(L)
###########
# b
###########
a1 = 1
a2 = 1
n = int(input('n = '))

print(a1, a2, end =' ')

for x in range (n - 2):
    a = a1 + a2
    a1 = a2
    a2 = a
    print(a, end = ' ')