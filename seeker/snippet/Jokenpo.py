#date: 2025-06-13T17:00:19Z
#url: https://api.github.com/gists/f3ba077e1d901601f70c67a8931b17b0
#owner: https://api.github.com/users/LollyTulina

import os
from random import randint

rodadas = 1
opcaoP1 = 99
opcaoP2 = 99
vitP1 = 0
empP1 = 0
vitP2 = 0
empP2 = 0
jogoInvalido = 0
nomeP1 = input("Boas vindas ao Jokenpô!\nQual é o seu nome?\n")
os.system('cls')
print("Olá, {}!\nO Jokenpô irá funcionar da seguinte maneira:\n- O jogo possui 2 modos, Player Vs Player"
" e Player vs Computador;\n- O jogador deverá escolher entre Pedra[1], Papel[2] e Tesoura[3];\n"
"- O jogo poderá ser encerrado a qualquer momento, basta digitar 9 no lugar da resposta;\n"
"- Ao final do jogo, o jogador poderá consultar o histórico da partida.\n".format(nomeP1))

modoJogo = int(input("Qual modo de jogo você gostaria de jogar?\n"
                     ">Player vs Player[1]\n>Player vs Computador[2]\n>"))
os.system('cls')
if modoJogo == 1:
    nomeP2 = input("Digite o nome do jogador 2: ")
os.system('cls')

while rodadas != 0:
    if modoJogo == 1:
        print(">Lembre-se: Pedra[1], Papel[2] e Tesoura[3], ou 9 para encerrar o jogo.")
        opcaoP1 = int(input(">Jogador 1: "))
        os.system('cls')
        opcaoP2 = int(input(">Jogador 2: "))
        os.system('cls')
        if (opcaoP1 == 1 and opcaoP2 == 3) or (opcaoP1 == 2 and opcaoP2 == 1) or (opcaoP1 == 3 and opcaoP2 == 2):
            print("{} venceu está rodada!".format(nomeP1))
            vitP1 += 1
        elif (opcaoP2 == 1 and opcaoP1 == 3) or (opcaoP2 == 2 and opcaoP1 == 1) or (opcaoP2 == 3 and opcaoP1 == 2):
            print("{} venceu está rodada!".format(nomeP2))
            vitP2 += 1
        elif opcaoP1 == 9 or opcaoP2 == 9:
            print("Jogo finalizado!")
            rodadas = 0
        elif opcaoP1 == opcaoP2:
             print("Empate.")
             empP1 += 1
             empP2 += 1
        else:
             print("Resposta inválida.")
    elif modoJogo == 2:
        escolha = ('Pedra', 'Papel', 'Tesoura')
        computador = randint(0, 2)
        nomeP2 = 'IA'
        print('''Lembre-se, suas opções são:
        [1] Pedra
        [2] Papel
        [3] Tesoura
        [9] Sair''')
        jogador = int(input('Digite sua escolha:')) -1
        if jogador == 8:
            print("Jogo finalizado!")
            rodadas = 0
        elif (jogador != 0) and (jogador != 1) and (jogador != 2):
            print('Resposta inválida')
        else:
            print('-*' * 16)
            print(f'O computador jogou {escolha[computador]}')
            print(f'O jogador jogou {escolha[jogador]}')
            print('-*' * 16)
            if computador == 0: #maquina jogou pedra
                if jogador ==0:
                    print('Empate')
                    empP1 += 1
                    empP2 += 1
                elif jogador == 1:
                    print('Você venceu!')
                    vitP1 += 1
                elif jogador == 2:
                    print('Você perdeu')
                    vitP2 += 1
            elif computador == 1: #maquina jogou papel
                if jogador ==0:
                    print('Você perdeu')
                    vitP2 += 1
                elif jogador == 1:
                    print('Empate')
                    empP1 += 1
                    empP2 += 1
                elif jogador == 2:
                    print('Você venceu!')
                    vitP1 += 1
            elif computador == 2: #maquina jogou tesoura
                if jogador ==0:
                    print('Você venceu!')
                    vitP1 += 1
                elif jogador == 1:
                    print('Você perdeu')
                    vitP2 += 1
                elif jogador == 2:
                    print('Empate')
                    empP1 += 1
                    empP2 += 1
    else:
        print('Resposta inválida.')
        rodadas = 0
        jogoInvalido = 2

if jogoInvalido != 2:
    print("\n>Histórico da última partida:\n"
    ">Jogador 1 [{}]:\n"
    "  Vitória(s): {}.\n"
    "  Empates(s): {}.\n"
    ">Jogador 2 [{}]:\n"
    "  Vitória(s): {}.\n"
    "  Empates(s): {}.\n".format(nomeP1, vitP1, empP1, nomeP2, vitP2, empP2))