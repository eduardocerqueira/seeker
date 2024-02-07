#date: 2024-02-07T16:58:17Z
#url: https://api.github.com/gists/2ed41a837036fcb59c73a2b6253caabd
#owner: https://api.github.com/users/mariocanul23

print('Bienvenido al sistema de los retos de Python \n')

while True:
    usuarios = int(input('Cuantos usuarios desea registrar: '))
    if usuarios > 0:
        break
    else:
        print('Por favor debe ingresar al menos a un usuario.')


for _ in range(usuarios):

    print('Le tomaremos sus datos para su registro')

    while True:
        primer_nombre = input('Escriba su primer nombre: ')
        segundo_nombre = input('Escriba su segundo nombre: ')
        if 5 <= len(primer_nombre) <= 50 and 5 <= len(segundo_nombre) <= 50:
            break
        else:
            print('La longitud debe estar entre 5 y 50 caracteres. Intenta nuevamente.')

    while True:
        apellido_paterno = input('Escriba su apellido paterno: ')
        apellido_materno = input('Escriba su apellido materno: ')
        if 5 <= len(apellido_paterno) <= 50 and 5 <= len(apellido_materno) <= 50:
            break
        else:
            print('La longitud debe estar entre 5 y 50 caracteres. Intenta nuevamente.')
    
    while True:
        correo_electronico = input('Escriba su correo electrónico: ')
        if 5 <= len(correo_electronico) <= 50:
            break
        else:
            print('La longitud debe estar entre 5 y 50 caracteres. Intenta nuevamente.')
    
    while True:
        telefono = int(input('Escriba su número de teléfono: '))
        if len(str(telefono)) == 10:
            break
        else:
            print('La longitud debe ser 10 caracteres. Intenta nuevamente.') 

    print('Hola '+ primer_nombre + ' ' + segundo_nombre + ' ' + apellido_paterno + ' ' + apellido_materno + ',')
    print('en breve recibirás un correo a ' + correo_electronico + '.')