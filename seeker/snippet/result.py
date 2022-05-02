#date: 2022-05-02T16:47:35Z
#url: https://api.github.com/gists/d31b18b2c47c6b11feffe4b26de96b85
#owner: https://api.github.com/users/miroslavovna

kol_vo = int(input("Введите количество билетов и нажмите enter: "))
print("Введите возраст для каждого участника и нажмите enter: ")
summa = 0

for i in range(kol_vo):
    vozrast = int(input("Возраст (полных лет): "))
    if vozrast < 18:
        summa += 0
    elif vozrast >= 25:
        summa += 1390
    else:
        summa += 990

if kol_vo < 3:
    itogo = summa
else:
    itogo = summa*0.9

print("К оплате: " , itogo)

