#date: 2024-04-26T16:37:59Z
#url: https://api.github.com/gists/8e62e28e18fb5431eb3cfd59f184fabd
#owner: https://api.github.com/users/MayeDeveloper

#Programa que hace la presentación del menu con los tipos de pizza de Harrys Pizza

print("Bienvenido a la pizzaria Harrys Pizza. \n Tipos de pizza\n\t1 - Vegetariana\n\t2 - No Vegetariana\
tipo = input("Introduce el número correpondiente al tipo de pizza que deseas: ")
"""Decision sobre el tipo de Piza"""
if tipo == "1":
    print("Seleccione el ingrediente de pizza vegetariana: \n\t 1 - Pimiento\n\t 2 - Tofu\n")
    ingrediente = input("Selecciones el codigo del ingrediente: ")
    print("Usted ha seleccionado una pizza vegetariana con mozarella, tomate y ", end="")
    if ingrediente == "1":
        print("Pimiento")
    else:
        print("Tofu")
else:
    print("Selecciona el ingrediente de pizza NO vegetariana: \n\t 1 - Peperoni\n\t 2 - Jamón\n\t 3 - Sal
    ingrediente = input("Seleccione el codigo del ingrediente: ")
print("Usted eligió una Pizza No vegetariana con mozarella, tomate y ", end="")
if ingrediente == 1:
    print("Peperoni")
elif ingrediente == 2:
    print("Jamón")
else:
    print("Salmón")
