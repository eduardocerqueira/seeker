#date: 2025-06-11T17:04:02Z
#url: https://api.github.com/gists/d001bece6a01de6c49e97d3fa4bb5a28
#owner: https://api.github.com/users/DmytroPin

# Якщо хоча б одна оцінка нижче 3, студент не допускається до іспиту.
# Якщо всі оцінки 4 і вище, студент допускається до іспиту з відзнакою.
# У всіх інших випадках студент допускається до іспиту.

grade1=int(input('Введіть оцінку з математики: '))
grade2=int(input('Введіть оцінку з української мови: '))
grade3=int(input('Введіть оцінку з інформатик: '))
grade4=int(input('Введіть оцінку з англійської мови: '))
if grade1<3 or grade2<3 or grade3<3 or grade4<3:
    print('Не допускаєтеся до іспиту')
elif grade1>=4 or grade2>=4 or grade3>=4 or grade4>=4:
    print("Допуск до іспиту з відзнакою")
else:
    print('Допуск до іспиту')