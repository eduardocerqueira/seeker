#date: 2023-08-30T17:00:14Z
#url: https://api.github.com/gists/aa220440ba50ebdb33f9a431128f6dfa
#owner: https://api.github.com/users/N1klausz

valor = 0  # guarda os valores das pizzas

print('Bem vindo a Pizzaria Lucas')
print('| Código |   Descrição   | Pizza Média - M | Pizza Grande - G(30% mais cara) |')
print('|   21   | Napolitana    |        R$ 20,00 |                         R$ 26,00|')
print('|   22   | Margherita    |        R$ 20,00 |                         R$ 26,00|')
print('|   23   | Calabresa     |        R$ 25,00 |                         R$ 32,50|')
print('|   24   | Toscana       |        R$ 30,00 |                         R$ 39,00|')
print('|   25   | Portuguesa    |        R$ 30,00 |                         R$ 39,00|')

while True:
    tam = input('Digite o tamanho da pizza, Media ou Grande (M/G): ')  # apenas M ou G serão aceitas
    if tam == 'M':  # aqui o programa espera q a letra M seja digitada para dar inicio
        cod = input(
            'Digite o codigo da pizza desejada: ')  # aqui apenas os codigos/numeros disponiveis no cardapio serão aceitos 21,22,23,24 e 25
        if cod == '21':
            print('Voce pediu uma pizza Napolitana Media')  # msg personalizada com o sabor escolhido
            valor += 20  # somatório dos pedidos valor = valor = x(valor da pizza escolhida)
        elif cod == '22':
            print('Voce pediu uma pizza Margherita Media')
            valor += 20
        elif cod == '23':
            print('Voce pediu uma pizza Calabresa Media')
            valor += 25
        elif cod == '24':
            print('Voce pediu uma pizza Toscana Media')
            valor += 30
        elif cod == '25':
            print('Voce pediu uma pizza Portuguesa Media')
            valor += 30
        else:
            print('Opção invalida')  # msg dada caso seja colocado numeros diferentes dos aceitos
            continue
        resposta = input(
            'Deseja algo mais? (s/n) ')  # msg dada ao fazer um pedido com sucesso, se deseja continuar ou encerrar
        if resposta == 's':
            continue  # se decidir por continuar a compra essa opção o levara ao inicio
        else:
            break  # caso queira terminar esta finalizara


    elif tam == 'G':  # opção para a pizza G
        cod = input('Digite o codigo da pizza desejada: ')
        if cod == '21':
            print('Voce pediu uma pizza Napolitana Grande')
            valor += 26
        elif cod == '22':
            print('Voce pediu uma pizza Margherita Grande')
            valor += 26
        elif cod == '23':
            print('Voce pediu uma pizza Calabresa Grande')
            valor += 32.5
        elif cod == '24':
            print('Voce pediu uma pizza Toscana Grande')
            valor += 39
        elif cod == '25':
            print('Voce pediu uma pizza Portuguesa Grande')
            valor += 39
        else:
            print('Opção invalida')
            continue
        resposta = input('Deseja algo mais? (s/n) ')
        if resposta == 's':
            continue
        else:
            break
    else:
        print('Opção inválida')  # caso digite algo que nao seja M ou G isso aparecera
        continue

print('O total da sua compra é de R${}'.format(valor))  # msg com o valor total da compra!
