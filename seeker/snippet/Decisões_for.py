#date: 2022-07-04T17:10:22Z
#url: https://api.github.com/gists/3c9c281c6cd9ab990b08892883b8563c
#owner: https://api.github.com/users/DaviGaldeano

tabuada=int(input("Digite um numero para exibir a tabuada:"))
print("Tabuada do número", tabuada)
for valor in range (1,11,1):
    print(str(tabuada) + "x" + str(valor) + ' = '+ str((tabuada*valor)))

