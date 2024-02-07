#date: 2024-02-07T16:55:06Z
#url: https://api.github.com/gists/4d6f682795a3683c59cdbf2fcf18d1ca
#owner: https://api.github.com/users/JoelVilc

def obtener_mensaje(mensaje, longitud_min, longitud_max):
    while True:
        entrada = input(mensaje)
        if longitud_min <= len(entrada) <= longitud_max:
            return entrada
        else:
            print(
                f'Error: La longitud debe estar entre {longitud_min} y {longitud_max} caracteres. Introduzca nuevamente.')


def obtener_tel(telf, long):
    while True:
        entrada = input(telf)
        if len(entrada) == 10:
            return entrada
        else:
            print(
                f'Error: La longitud debe ser de {long} caracteres. Introduzca nuevamente.')


cant_usuarios = int(input('Ingrese la cantidad de usuarios a registrar: '))
contador = 1

usuarios_list = []

while contador <= cant_usuarios:
    print('-------------------------------------')
    print(f'registrando el usuario número {contador}')

    nombre = obtener_mensaje('Ingrese sus nombres: ', 5, 50)
    apellidos = obtener_mensaje('Ingrese sus apellidos: ', 5, 50)
    correo = obtener_mensaje('Ingrese su correo electrónico: ', 5, 50)
    numero_telefono = obtener_tel('Ingrese su número de teléfono: ', 10)

    usuarios_dict = {
        'nombre': nombre,
        'apellidos': apellidos,
        'correo': correo,
        'numero_telefono': numero_telefono,
        'id': contador
    }

    usuarios_list.append(usuarios_dict)

    contador += 1

print(f'\nUsuarios:\n{usuarios_list}')