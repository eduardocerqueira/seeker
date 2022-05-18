#date: 2022-05-18T17:09:40Z
#url: https://api.github.com/gists/c32a64930cbc83e45da30640259333d3
#owner: https://api.github.com/users/Danniel30

import forca
import adivinhacao

def escolhe_jogo():
    print("*********************************")
    print("**************Escolha o seu jogo!")
    print("*********************************")

    print("(1) Forca (2) Adivinhação")

    jogo = int(input("Qual jogo?"))

    if(jogo == 1):
        print("Jogoando forca")
        forca.jogar()
    elif (jogo == 2):
        print("Jogando adivinhação")
        adivinhacao.jogar()

if (__name__ == "__main__"):
    escolhe_jogo()
