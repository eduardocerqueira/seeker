#date: 2023-06-06T17:04:01Z
#url: https://api.github.com/gists/238ec9d9ac11c99c0667f6e621d1f32b
#owner: https://api.github.com/users/ViniPolonio

def adicao(x, y):
    return x + y

def subtracao(x, y):
    return x - y

def multiplicacao(x, y):
    return x * y

def divisao(x, y):
    if y != 0:
        return x / y
    else:
        return "Erro: divisão por zero!"

def porcentagem(x, y):
    return x * (y / 100)

print("="*27)
print("  Bem-vindo à calculadora!")
print("="*27)
while True:
    print("""
Selecione a operação
[1] Adição")
[2] Subtração")
[3] Multiplicação")
[4] Divisão")
[5] Porcentagem")
[0] Sair""")


    escolha = input("Digite o número da operação desejada: ")

    if escolha == "0":
        print("Fechando a calculadora...")
        break

    if escolha not in ["1", "2", "3", "4", "5"]:
        print("Escolha inválida, tente novamente.")
        continue

    num1 = float(input("Digite o primeiro número: "))
    num2 = float(input("Digite o segundo número: "))

    if escolha == "1":
        resultado = adicao(num1, num2)
        print(f"A soma de {num1} + {num2} = {resultado}")
    elif escolha == "2":
        resultado = subtracao(num1, num2)
        print(f"A subtração de {num1} - {num2} = {resultado}")
    elif escolha == "3":
        resultado = multiplicacao(num1, num2)
        print(f"A multiplicação de {num1} * {num2} = {resultado}")
    elif escolha == "4":
        resultado = divisao(num1, num2)
        print(f"A divisão de {num1} / {num2} = {resultado}")
    elif escolha == "5":
        resultado = porcentagem(num1, num2)
        print(f"{num2}% de {num1} = {resultado}")