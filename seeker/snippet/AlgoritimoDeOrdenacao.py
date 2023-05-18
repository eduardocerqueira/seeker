#date: 2023-05-18T17:04:48Z
#url: https://api.github.com/gists/535adcbdc6e82533722d19853b6c06d4
#owner: https://api.github.com/users/dieguesmosken

#Algoritmo de ordenação escrito em Python sem usar os operadores de comparação < e >
import math
lista = [14, 25, 5, 19500, 36, 400, 99, 2, 10, 0, -1, -30]

for i in range(len(lista)-1):
    for j in range(len(lista)-i):
        maior = int(((lista[i]+lista[i+1])+(math.sqrt(math.pow((lista[i]-lista[i+1]), 2))))/2)
        lista.append(maior)
        if lista[i] != maior:
            lista.remove(lista[i+1])
        else:
            lista.remove(lista[i])

print(lista)