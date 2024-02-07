#date: 2024-02-07T16:55:52Z
#url: https://api.github.com/gists/cfd3fa97d39da4431b07911aeab79366
#owner: https://api.github.com/users/xavisoco

# RETO3 - estructuras de datos (tuplas)

lista_usr = []

# Parámetros de los datos
long_min = 5
long_max = 50
long_tel = 10
usr_reg = 0

cantidad_usr = int( input('Introduce el nombre de usuarios a registrar: '))
print('\n')

while usr_reg < cantidad_usr:

    usr_reg +=1
    print('REGISTRO ' + str(usr_reg))
    print('----------------------------------------------------')
    
    validado = False
    while not validado:
        nombre = input('Introduce el nombre: ')
        if len(nombre) >= long_min and len(nombre) <= long_max:
            validado = True
        else:
             print('ERROR en la longitud de los datos introducidos, es necesario volver a introducirlos.\n')

    validado = False
    while not validado:
            apellidos = input('Introduce los apellidos: ')
            if len(apellidos) >= long_min and len(apellidos) <= long_max:
                validado = True
            else:
                print('ERROR en la longitud de los datos introducidos, es necesario volver a introducirlos.\n')

    validado = False
    while not validado:
            telefono = input('Introduce el teléfono: ')
            if len(telefono) == long_tel:
                validado = True
            else:
                print('ERROR en la longitud de los datos introducidos, es necesario volver a introducirlos.\n')
                 
    validado = False
    while not validado:
            correo = input('Introduce el correo electrónico: ')
            if len(correo) >= long_min and len(correo) <= long_max:
                validado = True
            else:
                print('ERROR en la longitud de los datos introducidos, es necesario volver a introducirlos.\n')
             
    
    print('\nHola ',nombre ,' ',apellidos ,', en breve recibiras un correo a ',correo,'\n')
   
    # Añade a una tupla el nuevo usuario
    lista_usr.append ([usr_reg, nombre, apellidos, telefono, correo])

print('----------------------------------------------------\n')
print('...... Registro de usuarios finalizado. \n')

# Muestra el listado de usuarios guardados
print('----------------------------------------------------\n\n')
print('...... Listado de usuarios \n')

for reg in lista_usr:
    print(reg)
