#date: 2025-06-23T17:06:21Z
#url: https://api.github.com/gists/71d7748952d2894e1b6055b6ceb3baa4
#owner: https://api.github.com/users/LuisVk23

n1 = int(input('Digite um valor: '))
n2 = int(input('Digite outro valor: '))
operacao = input('Escolha a operação:(+,-,*,/)')

if operacao == '+':
    resultado = n1 + n2
    print(resultado)

elif operacao == '-':
    resultado = n1 - n2
    print(resultado)

elif operacao == '*':
    resultado = n1 * n2
    print(resultado)

elif operacao == '/':
    resultado = n1 / n2
    print(resultado)

else:
    print('operaçao invalida!')




