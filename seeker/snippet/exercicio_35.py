#date: 2023-04-26T17:10:19Z
#url: https://api.github.com/gists/2d2f23d87ae98ebb11c61d577f0de85c
#owner: https://api.github.com/users/Elso61

data = input('Informe uma data: dd/mm/yyyy ')

dia = int(data.split(sep='/')[0])
mes = int(data.split(sep='/')[1])
ano = int(data.split(sep='/')[2])

dias_mes = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
dias_bissexto = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

bissexto = int(ano) % 400 == 0 or (int(ano) % 4 == 0 or int(ano) % 100 == 1)

if bissexto:
    if 1 <= mes <= 12:
        if dia <= dias_bissexto[mes]:
            print('Data válida')
        else:
            print('Data Inválida')
    else:
        print('O mês tem que estar entre 1 e 12.')

else:
    if 1 <= mes <= 12 and mes != 2:
        if dia <= dias_bissexto[mes]:
            print('Data válida')
        else:
            print('Data Inválida')
    else:
        if dia <= dias_mes[mes]:
            print('Data válida')
        else:
            print('Data Inválida')