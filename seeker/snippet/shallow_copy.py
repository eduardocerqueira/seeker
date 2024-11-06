#date: 2024-11-06T16:54:28Z
#url: https://api.github.com/gists/683c67fd6404100e21690efdd2e102f0
#owner: https://api.github.com/users/joaomedeirosr

# Alterando objetos/valores de nível superior

lista_original1 = [[15],[30],[45]]
new_shallow_copy = lista_original1.copy()
lista_referencia = lista_original1
lista_original1[1] = 0


print(lista_original1)
print(new_shallow_copy)
print(lista_referencia)

#Output
#[[15],[0],[45]]
#[[15],[30],[45]]
#[[15],[0],[45]]