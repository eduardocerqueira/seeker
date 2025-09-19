#date: 2025-09-19T17:08:55Z
#url: https://api.github.com/gists/33999b96b3337e96b1a7899815109a82
#owner: https://api.github.com/users/beatrizdamasvalente-pixel

import random

palavras = ["programação", "computador", "python", "universidade", "algoritmo", "afterschool"]

HANGMAN = [
    """
     ------
     |    
     |
     |
     |
     |
    ---
    """,
    """
     ------
     |    O
     |
     |
     |
     |
    ---
    """,
    """
     ------
     |    O
     |    |
     |
     |
     |
    ---
    """,
    """
     ------
     |    O
     |   /|
     |
     |
     |
    ---
    """,
    """
     ------
     |    O
     |   /|\\
     |
     |
     |
    ---
    """,
    """
     ------
     |    O
     |   /|\\
     |   /
     |
     |
    ---
    """,
    """
     ------
     |    O
     |   /|\\
     |   / \\
     |
     |
    ---
    """,
]

def escolher_palavra():
    return random.choice(palavras)

def mostrar_palavra(palavra, letras_certas):
    return " ".join([letra if letra in letras_certas else "_" for letra in palavra])

def jogar():
    print("HANGMAN")
    palavra = escolher_palavra()
    letras_certas = set()
    letras_usadas = set()
    erros = 0
    max_erros = len(HANGMAN) - 1

    while erros < max_erros:
        print(HANGMAN[erros])
        print("\nPalavra:", mostrar_palavra(palavra, letras_certas))
        print("Letras utilizadas:", " ".join(sorted(letras_usadas)))
        
        tentativa = input("Qual a tua jogada? ").lower().strip()

        if len(tentativa) == 1: 
            if tentativa in letras_usadas:
                print("Já tentaste essa letra!")
                continue
            letras_usadas.add(tentativa)

            if tentativa in palavra:
                letras_certas.add(tentativa)
                if all(letra in letras_certas for letra in palavra):
                    print(f"Parabéns! A palavra era '{palavra}'. Ganhaste!")
                    return
            else:
                erros += 1
                print("Letra incorreta!")
        else:  
            if tentativa == palavra:
                print(f"Parabéns! A palavra era '{palavra}'. Ganhaste!")
                return
            else:
                erros += 1
                print("Palavra incorreta!")

    print(HANGMAN[erros])
    print(f"Game over! A palavra era '{palavra}'.")

if __name__ == "__main__":
    jogar()