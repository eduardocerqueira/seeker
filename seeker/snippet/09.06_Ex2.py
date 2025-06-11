#date: 2025-06-11T17:04:02Z
#url: https://api.github.com/gists/d001bece6a01de6c49e97d3fa4bb5a28
#owner: https://api.github.com/users/DmytroPin

#Користувач вводить із клавіатури номер місяця (1-12). Необхідно вивести на екран назву місяця.
#  Наприклад, якщо 1, то на екрані напис січень, 2 — лютий тощо.
month_numb=int(input('Введіть номер місяця: '))
if month_numb==1:
    print('Січень')
elif month_numb==2:
    print('Лютий')
elif month_numb==3:
    print('Березень')
elif month_numb==4:
    print('Квітень')
elif month_numb==5:
    print('Травень')
elif month_numb==6:
    print('Червень')
elif month_numb==7:
    print('Липень')
elif month_numb==8:
    print('Серпень')
elif month_numb==9:
    print('Вересень')
elif month_numb==10:
    print('Жовтень')
elif month_numb==11:
    print('Листопад')
elif month_numb==12:
    print('Грудень')
else:
    print('Нема місяця з таким номером!')
