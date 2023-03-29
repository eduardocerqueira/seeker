#date: 2023-03-29T17:54:03Z
#url: https://api.github.com/gists/d7363b04daacfef8f44d1cd06bd1fc80
#owner: https://api.github.com/users/americapolly

import math

def menu():
    print("Escolha uma opção:")
    print("1 - Adição")
    print("2 - Subtração")
    print("3 - Multiplicação")
    print("4 - Divisão")
    print("5 - Exponenciação")
    print("6 - Raiz quadrada")
    print("7 - Logaritmo")
    print("8 - Fatorial")
    print("9 - Sair")

def adicao():
    num1 = float(input("Digite o primeiro número: "))
    num2 = float(input("Digite o segundo número: "))
    resultado = num1 + num2
    print("Resultado: ", resultado)

def subtracao():
    num1 = float(input("Digite o primeiro número: "))
    num2 = float(input("Digite o segundo número: "))
    resultado = num1 - num2
    print("Resultado: ", resultado)

def multiplicacao():
    num1 = float(input("Digite o primeiro número: "))
    num2 = float(input("Digite o segundo número: "))
    resultado = num1 * num2
    print("Resultado: ", resultado)

def divisao():
    num1 = float(input("Digite o primeiro número: "))
    num2 = float(input("Digite o segundo número: "))
    resultado = num1 / num2
    print("Resultado: ", resultado)

def exponenciacao():
    num1 = float(input("Digite o número: "))
    num2 = float(input("Digite a potência: "))
    resultado = math.pow(num1, num2)
    print("Resultado: ", resultado)

def raiz_quadrada():
    num = float(input("Digite o número: "))
    resultado = math.sqrt(num)
    print("Resultado: ", resultado)

def logaritmo():
    num = float(input("Digite o número: "))
    base = float(input("Digite a base do logaritmo: "))
    resultado = math.log(num, base)
    print("Resultado: ", resultado)

def fatorial():
    num = int(input("Digite o número: "))
    resultado = math.factorial(num)
    print("Resultado: ", resultado)

while True:
    menu()
    opcao = int(input("Opção: "))

    if opcao == 1:
        adicao()
    elif opcao == 2:
        subtracao()
    elif opcao == 3:
        multiplicacao()
    elif opcao == 4:
        divisao()
    elif opcao == 5:
        exponenciacao()
    elif opcao == 6:
        raiz_quadrada()
    elif opcao == 7:
        logaritmo()
    elif opcao == 8:
        fatorial()
    elif opcao == 9:
        print("Encerrando...")
        break
    else:
        print("Opção inválida.")