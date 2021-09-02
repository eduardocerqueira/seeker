#date: 2021-09-02T17:14:46Z
#url: https://api.github.com/gists/d4c127388aea0721f29f259f1ab57e22
#owner: https://api.github.com/users/Jeanrh

import random
Valor_player =0
while True:
    pc_valor = random.randint(1, 10)
    Valor_player = int(input('Digite um valor de 1 a 10: '))
    if Valor_player != pc_valor:
        print('Numero incorreto!')
        print(f'Valor escolhido pela maquina foi {pc_valor}')
    elif Valor_player == pc_valor:
        print('Boa você acertou o numero!')
        print(f'Valor escolhido pela maquina foi {pc_valor}')
        break
print('Fim do programa')