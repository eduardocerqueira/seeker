#date: 2023-05-12T17:00:35Z
#url: https://api.github.com/gists/058a4c2319266e37660b4a28c2c0ce54
#owner: https://api.github.com/users/FabricioPaivaLima

operadores_aceitos = '+-*/='
lista_expressao = list()
lista_operador = list()

expressao = input('Informe a expressão: ').strip().lower()
expressao = expressao.replace(' ', '')
print(expressao)


for c in expressao:
    lista_expressao.append(c)
while True:
    for c in lista_expressao:
        if c == lista_expressao[1]:
            if c in operadores_aceitos:
                lista_operador.append(c)
        if c == lista_expressao[3]:
            if c in operadores_aceitos:
                lista_operador.append(c)
    if len(lista_operador) != 2:
        print('Erro nos operadores.')
    break
v1 = input(f'Informe o valor de {expressao[0]}, caso seja sua incognita digite X: ').lower().strip()
if v1.isnumeric():
    v1 = float(v1)

v2 = input(f'Informe o valor de {expressao[2]}, caso seja sua incognita digite X: ').lower().strip()
if v2.isnumeric():
    v2 = float(v2)

v3 = input(f'Informe o valor de {expressao[4]}, caso seja sua incognita digite X: ').lower().strip()
if v3.isnumeric():
    v3 = float(v3)

if lista_operador[0] == '=':

    if v1 == 'x':
        if lista_operador[1] == '+':
            print(v2 + v3)
        elif lista_operador[1] == '-':
            print(v2 - v3)
        elif lista_operador[1] == '*':
            print(v2 * v3)
        elif lista_operador[1] == '/':
            print(v2 / v3)
    elif v2 == 'x':
        if lista_operador[1] == '+':
            print(v1 - v3)
        elif lista_operador[1] == '-':
            print(v1 + v3)
        elif lista_operador[1] == '*':
            print(v1 / v3)
        elif lista_operador[1] == '/':
            print(v1 * v3)
    elif v3 == 'x':
        if lista_operador[1] == '+':
            print(v1 - v2)
        elif lista_operador[1] == '-':
            print(v1 + v2)
        elif lista_operador[1] == '*':
            print(v1 / v2)
        elif lista_operador[1] == '/':
            print(v1 * v2)
elif lista_operador[0] != '=':
    if v1 == 'x':
        if lista_operador[0] == '+':
            print(v2 + v3)
        elif lista_operador[0] == '-':
            print(v2 - v3)
        elif lista_operador[0] == '*':
            print(v2 * v3)
        elif lista_operador[0] == '/':
            print(v2 / v3)
    elif v2 == 'x':
        if lista_operador[0] == '+':
            print(v1 - v3)
        elif lista_operador[0] == '-':
            print(v1 + v3)
        elif lista_operador[0] == '*':
            print(v1 / v3)
        elif lista_operador[0] == '/':
            print(v1 * v3)
    elif v3 == 'x':
        if lista_operador[0] == '+':
            print(v1 + v2)
        elif lista_operador[0] == '-':
            print(v1 - v2)
        elif lista_operador[0] == '*':
            print(v1 * v2)
        elif lista_operador[0] == '/':
            print(v1 / v2)
