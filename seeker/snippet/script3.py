#date: 2024-02-07T17:03:14Z
#url: https://api.github.com/gists/30aaf43448f84e68f3ed353af24b91ef
#owner: https://api.github.com/users/TadekDuran

totalUsuarios = int(input('Indique la cantidad de usuarios que desea registrar: '))
usuariosRegistrados = 0
idUsuario = 0
listaIdUsuarios = []
while usuariosRegistrados < totalUsuarios:
    nombre = input('Ingrese su nombre/nombres (mínimo 5 caracteres, máximo 50): ')
    while len(nombre) < 5 or len(nombre) > 50:
        nombre = input('Formato inválido, el nombre debe contener mínimo 5 caracteres y máximo 50. Intente nuevamente: ')

    apellido = input('Ingrese su apellido/apellidos (mínimo 5 caracteres, máximo 50): ')
    while len(apellido) < 5 or len(apellido) > 50:
        apellido = input('Formato inválido, el apellido debe contener mínimo 5 caracteres y máximo 50. Intente nuevamente: ')

    numero_telefono = input('Ingrese su número telefónico (Debe tener 10 caracteres): ')
    while len(numero_telefono) != 10:
        numero_telefono = input('Formato inválido, el número telefónico debe contener 10 caracteres. Intente nuevamente: ')

    correo_electronico = input('Ingrese su correo electrónico (mínimo 5 caracteres, máximo 50): ')
    while len(correo_electronico) < 5 or len(correo_electronico) > 50:
        correo_electronico = input('Formato inválido, el correo debe contener mínimo 5 caracteres y máximo 50. Intente nuevamente: ')

    idUsuario += 1
    listaIdUsuarios.insert(len(listaIdUsuarios), idUsuario)

    print('Hola ' + nombre, apellido + ', en breve recibirá un correo a ' + correo_electronico + '. Se le asignó el ID =', idUsuario)
    usuariosRegistrados += 1
    
    if (totalUsuarios - usuariosRegistrados) == 1:
        print('Queda 1 usuario por registrar')
    elif (totalUsuarios - usuariosRegistrados == 0):
        print('Ya se registraron todos los usuarios. Los ID de usuarios asignados son:', listaIdUsuarios)
        print('Saliendo del programa...')
    else:
        print('Faltan registrar', totalUsuarios - usuariosRegistrados,  'usuarios.')