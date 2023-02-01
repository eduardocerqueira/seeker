#date: 2023-02-01T16:55:52Z
#url: https://api.github.com/gists/9cdc8b3114da23d2b75ef313f6172056
#owner: https://api.github.com/users/sybery1980

import random

def showMatrix():
    return f'{matrix[0]} {matrix[1]} {matrix[2]} \n{matrix[3]} {matrix[4]} {matrix[5]} \n{matrix[6]} {matrix[7]} {matrix[8]} \n'

def showMatrix():
    print(matrix[0],matrix[1],matrix[2])
    print(matrix[3],matrix[4],matrix[5])
    print(matrix[6],matrix[7],matrix[8])
    print('\n')
def checkWin():
    if matrix[0] == matrix[1] == matrix[2]:
        print(f'Победили {matrix[0]}')
        return 1
    elif matrix[3] == matrix[4] == matrix[5]:
        print(f'Победили {matrix[3]}')
        return 1
    elif matrix[6] == matrix[7] == matrix[8]:
        print(f'Победили {matrix[6]}')
        return 1
    elif matrix[0] == matrix[3] == matrix[6]:
        print(f'Победили {matrix[0]}')
        return 1
    elif matrix[1] == matrix[4] == matrix[7]:
        print(f'Победили {matrix[1]}')
        return 1
    elif matrix[2] == matrix[5] == matrix[8]:
        print(f'Победили {matrix[2]}')
        return 1
    elif matrix[0] == matrix[4] == matrix[8]:
        print(f'Победили {matrix[0]}')
        return 1
    elif matrix[2] == matrix[4] == matrix[6]:
        print(f'Победили {matrix[2]}')
        return 1
    else: return 0
matrix = ['X', 'X', 3, 'X', 5, 6, 7, 8, 9]



def player(text):
    while True:
        try:
             number = text                   #int(input("Введите номер ячейки, чтобы поставить крестик: "))
             if (matrix[number-1] != 'X') and (matrix[number-1] != 'O'):
                matrix[number-1] = 'X'
                break
             else:
                print("Неверный ввод. Ячейка занята")
        except:
            print("Неверный ввод")

def comp(matrix):
    for i in range(0,len(matrix)):
        if (matrix[i] != 'X') and matrix[i] != 'O':
            matrix[i] = 'O'
            return