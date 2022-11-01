#date: 2022-11-01T17:21:51Z
#url: https://api.github.com/gists/dcd0b05a83b06e25e211265c82841636
#owner: https://api.github.com/users/Galinaurievna

ticket = int(input("Введите количество билетов:\n"))
lst_age = []
sum = 0
for i in range(ticket):
    age = int(input("Введите возраст всех участников:\n"))
    lst_age.append(age)
    if 25 <= age:
        sum += 1390
    elif 18 <= age < 25:
        sum += 990
    else:
        sum += 0
if ticket > 3:
    sum = sum*0.9
print(sum)