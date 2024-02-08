#date: 2024-02-08T16:47:37Z
#url: https://api.github.com/gists/69a5bd8d0c6b24f4b0575c1a5ceb4e3f
#owner: https://api.github.com/users/btocarmona2021

nuevos_registros = int(input('ingrese la cantidad de usuarios a registrar: '))
registrados = 1
user_registrados = []
minimo = 5
maximo = 50
cant_caracteres = 0
telefono_valido = 0
while registrados <= nuevos_registros:
    telefono_valido = 0
    while cant_caracteres < minimo or cant_caracteres > maximo:
        print('debe ingresar un nombre con 5 o más carácteres y menor a 50 carácteres ')
        nombre = input('Ingrese su nombre por favor ')
        for caracteres in nombre:
            cant_caracteres += 1

    cant_caracteres = 0

    while cant_caracteres < minimo or cant_caracteres > maximo:
        print('debe ingresar un apellido con 5 o más carácteres y menor a 50 carácteres ')
        apellido = input('Ingrese su apellido por favor ')
        for caracteres in apellido:
            cant_caracteres += 1

    cant_caracteres = 0

    while cant_caracteres < minimo or cant_caracteres > maximo:
        print('debe ingresar un correo electrónico con 5 o más carácteres y menor a 50 carácteres ')
        correo_electronico = input('Ingrese su correo electrónico por favor ')
        for caracteres in correo_electronico:
            cant_caracteres += 1

    cant_caracteres = 0

    while telefono_valido != 10:

        print('debe ingresar un teléfono con 10 dígitos ')
        telefono = input('facilitenos su número telefónico ')
        for caracteres in telefono:
            telefono_valido += 1

    user_registrados.append('User-' + str(registrados))


    print(
        'Hola ' + nombre + ' ' + apellido + ', en breve nos comunicaremos contigo al teléfono ' + telefono + ' ,o bien recibirás un correo a ' + correo_electronico)
    registrados += 1
    print('---------------------------------------------------')

for registros in user_registrados:
    print('Los registros nuevos tienen la id siguiente: ' + registros)
