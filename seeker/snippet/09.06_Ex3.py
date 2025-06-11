#date: 2025-06-11T17:04:02Z
#url: https://api.github.com/gists/d001bece6a01de6c49e97d3fa4bb5a28
#owner: https://api.github.com/users/DmytroPin

#Користувач вводить суму покупки і свій вік. Програма обчислює знижку:
#Якщо вік менше 18, знижка 10%.
#Від 18 до 60 років — знижка 5%.
#Якщо вік більше 60 років — знижка 15%. 
# Програма виводить підсумкову суму з урахуванням знижки.

price =float(input('Введіть ціну товару: '))
age = int(input('Введіть ваш вік: '))
if age<=18 and age > 0:
    print('Знижка 10%')
    print('Ціна товару: ',price*0.1 )
elif age>18 and age<=60:
    print('Знижка 5%')
    print('Ціна товару: ', price*0.05)
elif age<=0:
    print('Народись спочатку!')
elif age>=122:
    print('Мертвим знижок не даємо!')
else:
    print("Знижка 15%")
    print("Ціна товару: ", price*0.15)