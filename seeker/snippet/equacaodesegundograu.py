#date: 2023-03-30T16:56:11Z
#url: https://api.github.com/gists/cfbc65901cf651fde0add784d4096f11
#owner: https://api.github.com/users/jaymestech

'''
1)Solicitar o valor do coeficiente a
2)Solicitar o valor do coeficiente b
3)Solicitar o valor do coeficiente c
4)Econtrar o valor do delta aplicando na fórmula os coeficientes recebidos
5)Se o delta >0, então:
5.1) Calcule a raiz de x1
5.2) Calcule a raiz de x2
5.3) Mostrar x1 e x2
6) Se o delta = 0, então:
6.1) Calcule a raiz x
6.2)Mostrar x
7) Se o delta < 0, então
7.1) Mostrar "Não existem raízes reais para a equação!"

'''


import math

a = float(input("Qual o valor de a?: "))
b = float(input("Qual o valor de b?: "))
c = float(input("Qual o valor de c?: "))


delta = b**2 - (4*a*c)

if delta < 0:
    print("Essa equação não possui raizes reais")
else:
    if delta == 0:
        print("Esta função possui apenas uma raiz")
        raiz = (b+ math.sqrt(delta))/a
        print("A raiz dessa equação é", raiz)
    else:
        print("Esta função possui duas raizes")
        raiz_1 = (-b + math.sqrt(delta)) / 2*a]
        raiz_2 = (-b - math.sqrt(delta)) / 2*a
        print("As raizes da equação são: ", raiz_1 , "e", raiz_2)

