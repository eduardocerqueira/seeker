#date: 2022-03-28T17:13:21Z
#url: https://api.github.com/gists/8e445f331fa9af719c5a60559ebbdd14
#owner: https://api.github.com/users/jmrocha88

import json
import random
import datetime



def jogar(secret = random.randint(1, 30), game_over = False, tentativa = 1, wrong_guesses = []):
    texto_1 = "Ainda não foi desta, tenta um valor mais alto:\n"
    texto_2 = "Ainda não foi desta, tenta um valor mais baixo:\n"
    with open("score.json", "r") as ficheiro_resultados:
        lista_resultados = json.loads(ficheiro_resultados.read())

    def guess_igual_secret(lista_resultados):
        print("Parabéns acertaste! O número secreto é: " + str(secret))
        if len(lista_resultados) <= 10 or tentativa < lista_resultados[9]['Tentativas']:
            nome = input("\nInsere o teu nome para a tabela de resultados:\n")
            lista_resultados.append(
                {"Tentativas": tentativa, "Data": str(datetime.datetime.now()), "Nome": nome, "Secret": secret,
                 "Erradas": wrong_guesses})
            lista_resultados = sorted(lista_resultados, key=lambda k: (k["Tentativas"], k['Data']))
            if len(lista_resultados) > 10:
                lista_resultados.pop(10)
            with open("score.json", "w") as ficheiro_resultados:
                ficheiro_resultados.write(json.dumps(lista_resultados))
        print("\n***Top 10 dos Melhores Resultados***\n")
        for top_10 in lista_resultados:
            print(str(top_10["Nome"]) + " - " + str(top_10["Tentativas"]) + " tentativas - em " + top_10.get(
                "Data") + " - o número secreto foi " + str(top_10["Secret"]) + " - errou com os números " + str(
                top_10["Erradas"]))

    def guess_diferente_secret(texto):
        return print(texto)

    print("\n***Lucky number game!***\n")

    while game_over == False:
        guess = int(input("Tentativa nº " + str(tentativa) + ". Insere um número de 1 a 30:\n"))
        if guess == secret:
            guess_igual_secret(lista_resultados)
            game_over = True
        elif guess < secret:
            guess_diferente_secret(texto_1)
            wrong_guesses.append(guess)
            tentativa += 1
        elif guess > secret:
            guess_diferente_secret(texto_2)
            wrong_guesses.append(guess)
            tentativa += 1
        if tentativa > 5:
            game_over = True
            print("---Perdeste o jogo!---")
    return tentativa, game_over, wrong_guesses

jogar()