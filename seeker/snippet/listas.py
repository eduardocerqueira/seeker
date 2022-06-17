#date: 2022-06-17T16:54:31Z
#url: https://api.github.com/gists/b396452f95174a370cc271f36a4cc5ff
#owner: https://api.github.com/users/Flaviolsantos

from random import randint

vendas = [
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
]
if __name__ == '__main__':

    for x in range(0, 2):
        for y in range(0, 4):
            for z in range(0, 3):
                vendas[x][y][z] = randint(10, 99)

    print(vendas)

    totalgasolina = 0
    for y in range(0,4):
        for z in range(0,3):
            totalgasolina += vendas[0][y][z]
            z += 1
        y += 1
    print(f'totalgasolina ={totalgasolina}')

    totalgasolio = 0
    for y in range(0,4):
        for z in range(0,3):
            totalgasolio += vendas[1][y][z]
            z += 1
        y += 1
    print(f'totalgasolio ={totalgasolio}')

    totalhomens = 0
    for x in range(0,2):
        for z in range(0,3):
            totalhomens = vendas[x][0][z]
            z += 1
        x += 1
    print(f'totalhomens = {totalhomens}')

    totalmulheres = 0
    for x in range(0,2):
        for z in range(0,3):
            totalmulheres = vendas[x][1][z]
            z += 1
        x += 1
    print(f'totalmulheres = {totalmulheres}')

    totalcriancas = 0
    for x in range(0,2):
        for z in range(0,3):
            totalcriancas = vendas[x][2][z]
            z += 1
        x += 1
    print(f'totalcriancas = {totalcriancas}')

    totalanimais = 0
    for x in range(0,2):
        for z in range(0,3):
            totalanimais = vendas[x][3][z]
            z += 1
        x += 1
    print(f'totalanimais = {totalanimais}')

    totalocidental = 0
    for x in range(0,2):
        for y in range(0,4):
            totalocidental += vendas[x][y][1]
            y += 1
        x += 1
    print(f'totalocidental = {totalocidental}')

    totalcentral = 0
    for x in range(0,2):
        for y in range(0,4):
            totalcentral += vendas[x][y][2]
            y += 1
        x += 1
    print(f'totalcentral = {totalcentral}')

    totaloriental = 0
    for x in range(0,2):
        for y in range(0,4):
            totaloriental += vendas[x][y][1]
            y += 1
        x += 1
    print(f'totaloriental = {totaloriental}')






