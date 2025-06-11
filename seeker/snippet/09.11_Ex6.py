#date: 2025-06-11T17:04:02Z
#url: https://api.github.com/gists/d001bece6a01de6c49e97d3fa4bb5a28
#owner: https://api.github.com/users/DmytroPin

# Якщо вік автомобіля менше 3 років і пробіг до 30000 км — "Автомобіль у відмінному стані".
# Якщо вік до 10 років і пробіг до 100000 км — "Автомобіль у хорошому стані".
# В інших випадках — "Автомобіль потребує перевірки".

car_age=int(input('Введіть вік авто: '))
car_mileage=int(input('Введіть пробіг авто: '))
if car_age<3 and car_mileage<=30000:
    print('Автомобіль у відмінному стані')
elif car_age<=10 and car_mileage<=100000:
    print('Автомобіль у хорошому стані')
else:
    print('Автомобіль потребує перевірки')