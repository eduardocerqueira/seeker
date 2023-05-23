#date: 2023-05-23T16:58:53Z
#url: https://api.github.com/gists/1083e7deafea2e7ade415863c1ca021b
#owner: https://api.github.com/users/DanilLoban

# Задача номер один
def sum(a, b, c):
    return (a +b +c)
print(sum(1, 2, 3))

# Задача вторая
Petr = int(input("Write numbers: "))
Sergey = Petr
Katerina = 2 * (Petr + Sergey)
Summa = Petr + Sergey + Katerina
print(Summa)

# Задача третья
tiket = input("Write number: ")
number1 = int(tiket[0]) + int(tiket[1]) + int(tiket[2])
number2 = int(tiket[3]) + int(tiket[4]) + int(tiket[5])
if number1 == number2:
    print("Yes")
else:
    print("No")

# Задача четвертая
n = int(input("Write number one parth: "))
m = int(input("Write number two parth: "))
k = int(input("Write number %: "))
if k % n == 0 or k % m == 0:
    print('Yes')
else:
    print('No')