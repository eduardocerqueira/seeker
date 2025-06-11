#date: 2025-06-11T17:04:02Z
#url: https://api.github.com/gists/d001bece6a01de6c49e97d3fa4bb5a28
#owner: https://api.github.com/users/DmytroPin

#Користувач вводить оцінки з трьох предметів (від 1 до 5).
#  Програма перевіряє, чи є серед них хоча б одна "двійка".
#  Якщо так, виводиться повідомлення "Незадовільно".
#  Якщо всі оцінки 4 і вище, виводиться "Відмінно".

grade1 = int(input('Введіть оцінку з математики: '))
grade2 = int(input('Введіть оцінку з англійської мови: '))
grade3 = int(input('Введіть оцінку з інформатики: '))
if grade1==2 or grade2==2 or grade3==2:
    print('Незадовільно!')
elif grade1>=4 and grade2>=4 and grade3>=4:
    print('Відмінно')