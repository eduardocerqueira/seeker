#date: 2023-06-23T17:05:33Z
#url: https://api.github.com/gists/92cd1e1f84db040678a04710e077d014
#owner: https://api.github.com/users/Jurodart

##Faça um programa que inicialize uma lista com vários números diferentes,
##depois disso, solicite ao usuário um número, verifique se o número está
##ou não na lista e exiba uma mensagem notificando o usuário do resultado.

lista=[2,4,6,8,10]
num=int(input("escolha um número\n"))

if (num in lista):
    print(num,"está na lista")
else:
    print(num,"não está na lista")
##ou fazer assim:
lista=[2,4,6,8,10]
num=int(input("escolha um número\n"))

if(num in lista):
    print(f"{num} está na lista")
else:
    print(f"{num} não está na lista")