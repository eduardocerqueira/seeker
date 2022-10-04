#date: 2022-10-04T17:30:42Z
#url: https://api.github.com/gists/6ac84e23986b186b5dcedf5cef078e5c
#owner: https://api.github.com/users/Valeriya4

#УВВ-111
#Хилинская Валерия
#Практическая работа 2. Магазин фруктов
#Магазин фруктов в будние дни продает по следующим ценам:
#Фрукт: banana apple orange grapefruit kiwi pineapple grapes 
#Стоимость/кг: 2.50 1.20 0.85 1.45 2.70 5.50 3.85
#В субботу и воскресенье магазин продает по более высоким ценам:
#Фрукт: banana apple orange grapefruit kiwi pineapple grapes 
#Стоимость/кг: 2.70 1.25 0.90 1.60 3.00 5.60 4.20
#Напишите программу, которая считывает из консоли
#фрукт (banana / apple / orange / grapefruit / kiwi / pineapple / grapes),
#день недели (Monday / Tuesday / Wednesday / Thursday / Friday / Saturday / Sunday),
#количество кг (вещественное число), и рассчитывает стоимость в соответствии с 
#ценами в таблицах выше. Выведите результат, округленный до 2 цифр 
#после точки. Если пользователь ввел неправильный день недели или 
#неправильное название фрукта, выведите «error»
# fruit - фрукт, который пользователь хочет найти
# day - день недели, который пользователь вводит самостоятельно
# quan - количество желаемого фрукта в кг, вводимое пользователем самостоятельно
# bb - стоимость бананов в будний день
# bv - стоимость бананов в выходной день
# ab - стоимость яблок в будний день
# av - стоимость яблок в выходной день
# ob - стоимость апельсинов в будний день
# ov - стоимость апельсинов в выходной день
# grb - стоимость грейпфрута в будний день
# grv - стоимость грейпфрута в выходной день
# kb - стоимость киви в будний день
# kv - стоимость киви в выходной день
# pb - стоимость ананаса в будний день
# pv - стоимость ананаса в выходной день
# gb - стоимость винограда в будний день
# gv - стоимость винограда в выходной день

print("Фрукты в наличии: banana, apple, orange, grapefruit, kiwi, pineapple, grapes") 
fruit = input("Введите фрукт, который хотите преобрести ")
day = input("Введите день недели с большой буквы ")
quan = float(input("Введите колличество желаемых фруктов в кг "))
if fruit == "banana" or fruit == "apple" or fruit == "orange" or fruit == "grapefruit" or fruit == "kiwi" or fruit == "pineapple" or fruit == "grapes":
    if fruit == "banana":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            bb = 2.50 * quan
            print(float('%.2f' % bb))
        elif day == "Saturday" or day == "Sunday":
            bv = 2.70 * quan
            print(float('%.2f' % bv))
        else:
            print("error")
    if fruit == "apple":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            ab = 1.20 * quan
            print(float('%.2f' % ab))
        elif day == "Saturday" or day == "Sunday":
            av = 1.25 * quan
            print(float('%.2f' % av))
        else:
            print("error")
    if fruit == "orange":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            ob = 0.85 * quan
            print(float('%.2f' % ob))
        elif day == "Saturday" or day == "Sunday":
            ov = 0.90 * quan
            print(float('%.2f' % ov))
        else:
            print("error")
    if fruit == "grapefruit":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            grb = 1.45 * quan
            print(float('%.2f' % grb))
        elif day == "Saturday" or day == "Sunday":
            grv =  1.60 * quan
            print(float('%.2f' % grv))
        else:
            print("error")
    if fruit == "kiwi":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            kb = 2.70 * quan
            print(float('%.2f' % kb))
        elif day == "Saturday" or day == "Sunday":
            kv = 3.00 * quan
            print(float('%.2f' % kv))
        else:
            print("error")
    if fruit == "pineapple":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            pb = 5.50 * quan
            print(float('%.2f' % pb))
        elif day == "Saturday" or day == "Sunday":
            pv = 5.60 * quan
            print(float('%.2f' % pv))
        else:
            print("error")
    if fruit == "grapes":
        if day == "Monday" or day == "Tuesday" or day == "Wednesday" or day == "Thursday" or day == "Friday":
            gb = 3.85 * quan
            print(float('%.2f' % gb))
        elif day == "Saturday" or day == "Sunday":
            gv = 4.20 * quan
            print(float('%.2f' % gv))
        else:
            print("error")
else:
    print("error")