#date: 2025-04-11T16:44:41Z
#url: https://api.github.com/gists/bbc10148d0583f38ad14ed9122f0662e
#owner: https://api.github.com/users/batmxia

import random

def jogo_adivinhacao():
    print("\n--- Jogo da Adivinhacao ---")
    print("Estou pensando em um numero entre 1 e 100. Tente adivinhar!")
    
    numero_secreto = "**********"
    tentativas = 0
    max_tentativas = 10
    
    while tentativas < max_tentativas:
        try:
            palpite = int(input("\nSeu palpite: "))
        except ValueError:
            print("Por favor, digite um número válido!")
            continue
        
        tentativas += 1
        
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"l "**********"p "**********"i "**********"t "**********"e "**********"  "**********"< "**********"  "**********"n "**********"u "**********"m "**********"e "**********"r "**********"o "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"o "**********": "**********"
            print(f"ERROU! O número é MAIOR que {palpite}. Tentativas restantes: {max_tentativas - tentativas}")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"p "**********"a "**********"l "**********"p "**********"i "**********"t "**********"e "**********"  "**********"> "**********"  "**********"n "**********"u "**********"m "**********"e "**********"r "**********"o "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"o "**********": "**********"
            print(f"ERROU! O número é MENOR que {palpite}. Tentativas restantes: {max_tentativas - tentativas}")
        else:
            print(f"\n✨ PARABÉNS! Você acertou em {tentativas} tentativas! O número era {numero_secreto}.")
            break
        
        # Mensagem da psicóloga após 3 tentativas
        if tentativas == 3:
            print("(Psicóloga falando: Respire fundo e tente padrões!)")
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"l "**********"p "**********"i "**********"t "**********"e "**********"  "**********"! "**********"= "**********"  "**********"n "**********"u "**********"m "**********"e "**********"r "**********"o "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"o "**********": "**********"
        print(f"\n💔 Fim de jogo! O número era {numero_secreto}.")

# Inicia o jogo
if __name__ == "__main__":
    jogo_adivinhacao()