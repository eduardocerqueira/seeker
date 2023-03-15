#date: 2023-03-15T17:07:52Z
#url: https://api.github.com/gists/736885dbfe1f4db260bb56e9de4d8f50
#owner: https://api.github.com/users/asman0

import random
print('Угадайте число от 1 до 100')
UnknowNumber=int(random.randrange(1,100,1))
print('Введи число')
UserNumber = int(input())
while UserNumber!=UnknowNumber:
    if UserNumber > UnknowNumber:
        print("Загаданное число меньше, введи другое")
        UserNumber = int(input())
    if UserNumber < UnknowNumber:
        print("Загаданное число больше, введи другое")
        UserNumber = int(input())
    if UserNumber==UnknowNumber:
        print('Да, верно, это число '+ str(UnknowNumber))
