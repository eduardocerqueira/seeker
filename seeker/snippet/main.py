#date: 2022-03-07T17:15:43Z
#url: https://api.github.com/gists/02da6ae450868a9e19e846973bc7efc0
#owner: https://api.github.com/users/Aleksegorov92

#Реализовать вывод информации о промежутке времени в зависимости от его продолжительности duration в секундах:
#до минуты: <s> сек;
#до часа: <m> мин <s> сек;
#до суток: <h> час <m> мин <s> сек;
#* в остальных случаях: <d> дн <h> час <m> мин <s> сек.
duration = int(input("Введите время в секундах: "))
s = duration
d = duration // 86400
h = duration % 86400 // 3600
m = duration % 86400 % 3600 // 60
s = duration % 86400 % 3600 % 60
if 86400 > duration >= 3600:
    print(h, "час", m, "мин", s, "сек")
elif 3600 > duration >= 60:
    print(m, "мин", s, "сек")
elif duration < 60:
    print(s, "сек")
else:
    duration > 86400
    print(d, "дн", h, "час", m, "мин", s, "сек")
