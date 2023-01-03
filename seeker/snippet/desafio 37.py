#date: 2023-01-03T16:40:46Z
#url: https://api.github.com/gists/2e7ebf0167628f984e8ea333d4002851
#owner: https://api.github.com/users/levilyn

print('\033[1;35m CONVERSOR DE NÚMEROS\033[0;37m')

numero = int(input('digite um numero inteiro qualquer para converter: '))
conversor = int(input('ok, agora escolha qual a base de conversao!\n'
                      'tecle 1 para binário\n'
                      'tecle 2 para octal\n'
                      'tecle 3 para hexadecimal\n'
                      'conversao escolhida: '))

if conversor == 1:
    binario = bin(numero)
    print('você escolheu o conversor binário, o número {} em binário é {}'.format(numero, binario))

elif conversor == 2:
    octal = oct(numero)
    print('você escolheu o conversor octal, o número {} em octal é {}'.format(numero, octal))


else:
    hexa = hex(numero)
    print('você escolheu o conversor hexadecimal, o número {} em hexadecimal é {}' .format(numero, hexa))
