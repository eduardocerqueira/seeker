#date: 2022-05-06T17:03:17Z
#url: https://api.github.com/gists/d5c5baf939ab21da3cea7aaab03316ba
#owner: https://api.github.com/users/Tiago-Fraga-1986

#Desafio 005 - Crie um programa que leia um número inteiro e informe seu sucessor e seu antecessor:
n = int(input('informe um número (qualquer número!) '))
s = n + 1
a = n - 1

print('você escolheu o número {}. O sucessor de {} é {} e o antecessor é {}.'.format(n, n, s, a))
#pode se substituir as variáveis "s" e "a" pelas operações dentro do .format, da seguinte forma:
#print('você escolheu o número {}. O sucessor de {} é {} e o antecessor é {}.'.format(n, n, (n+1), (n-1)))