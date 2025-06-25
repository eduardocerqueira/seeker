#date: 2025-06-25T17:14:48Z
#url: https://api.github.com/gists/58889beddfe859013d6c6ef4193ad63d
#owner: https://api.github.com/users/Dmytro-Pin

import random
robo=random.randint(1,100)
turns=0
while True:
    user=int(input('Введіть число в діапазоні від 1 до 100(для завершення - 0): '))
    if robo==user:
                print(f'Ви вгадали з {turns} спроби!')
                break

    elif user>robo:
        print("Бери меньше")
        turns+=1
    elif user==0:
          print('Завершення гри☹️')
          print(f'Ви спробували {turns} разів')
    else:
        print('Бери більше')
        turns+=1
