#date: 2024-05-29T17:11:36Z
#url: https://api.github.com/gists/6087d349388a0c5fde3ebb483b4039df
#owner: https://api.github.com/users/Tg7vg

import numpy as np

while True:
    try:
        QuantasNotas = int(input("Quantas notas do aluno são? "))
        if QuantasNotas < 2:
            print("Por favor, insira duas ou mais notas.")
        else:
            break
    except ValueError:
        print("Por favor, insira um número.")

notas = []
p = 1
x = 1

def n():
    while True:
        try:
            valor = float(input(f"Sua {x}° nota: "))
            if valor < 0:
                print("Por favor, insira um valor utilizável.")
            else:
                break
        except ValueError:
            print("Por favor, insira uma nota.")
    notas.append(valor)

while p <= QuantasNotas:
    n()
    p = p + 1
    x = x + 1
media =  np.mean(notas)

print(f"A média é: {media:.2f}")

input("Aperte Enter para sair...")