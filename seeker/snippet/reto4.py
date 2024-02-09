#date: 2024-02-09T16:45:08Z
#url: https://api.github.com/gists/207474e00c58bd5d969891e3df843dce
#owner: https://api.github.com/users/AdryEsc

identificadores = []
total_usuarios = []
continuar = 's'
opcion = 0

while opcion != 5:
    print('***MENU DEL PROGRAMA***')
    print('1 - INGRESAR NUEVO USUARIO')
    print('2 - LISTAR ID DE TODOS LOS USUARIOS')
    print('3 - BUSCAR UN USUARIO A TRAVES DE SU ID (Identificador)')
    print('4 - EDITAR INFORMACION DE UN USUARIO')
    print('5 - FINALIZAR Y SALIR DEL PROGRAMA')
    print(' ')
    opcion = int(input('Ingrese la opcion de la tarea a realizar: '))


    if opcion == 1:     # Ingreso de nuevo usuario
        print('***REGISTRO DE USUARIOS***')
        print('')

        cantidad_usuarios = int(input('Ingrese la cantidad de usuarios a registrar: '))
        print(' ')
        cantidad_iteraciones = 0
        LONGITUD_MINIMA = 5
        LONGITUD_MAXIMA = 50
        NUMERO_TELEFONO = 10
        id_usuario = 0
        # identificadores = []

        while cantidad_usuarios > 0 and cantidad_iteraciones < cantidad_usuarios:
            print('---Registro de nuevo usuario---')
            nombre = str(input('Ingrese el nombre: '))
            contador1 = 0
            for caracter in nombre:
            	contador1 = contador1 + 1
            while contador1 < LONGITUD_MINIMA or contador1 > LONGITUD_MAXIMA:
            		print('El nombre debe tener una longitud minima de 5 caracteres y una longitud maxima de 50 caracteres.')
            		nombre = str(input('Por favor, vuelva a ingresar un nombre valido:'))
            		contador1 = 0
            		for caracter in nombre:
            			contador1 = contador1 + 1

            		
            apellido = str(input('Ingrese el apellido: '))
            contador2 = 0
            for caracter in apellido:
            	contador2 = contador2 + 1
            while contador2 < LONGITUD_MINIMA or contador2 > LONGITUD_MAXIMA:
            		print('El apellido debe tener una longitud minima de 5 caracteres y una longitud maxima de 50 caracteres.')
            		apellido = str(input('Por favor, vuelva a ingresar un apellido valido:'))
            		contador2 = 0
            		for caracter in apellido:
            			contador2 = contador2 + 1


            telefono = input('Ingrese el telefono: ')
            contador3 = 0
            for numero in telefono:
            	contador3 = contador3 + 1
            while contador3 != 10:
            		print('El telefono debe tener 10 numeros.')
            		telefono = input('Por favor, vuelva a ingresar un telefono valido:')
            		contador3 = 0
            		for numero in telefono:
            			contador3 = contador3 + 1


            correo = str(input('Ingrese el correo: '))
            contador4 = 0
            for caracter in correo:
            	contador4 = contador4 + 1
            while contador4 < LONGITUD_MINIMA or contador4 > LONGITUD_MAXIMA:
            		print('El correo debe tener una longitud minima de 5 caracteres y una longitud maxima de 50 caracteres.')
            		correo = str(input('Por favor, vuelva a ingresar un correo valido:'))
            		contador4 = 0
            		for caracter in nombre:
            			contador4 = contador4 + 1

            cantidad_iteraciones = cantidad_iteraciones + 1

            print('Hola ' + nombre + ' ' + apellido + ',' + ' en breve recibiras un correo a ' + correo)
            print(' ')

            id_usuario = id_usuario + 1

            #usuarios = [id_usuario, nombre, apellido, telefono, correo]

            identificadores.append(id_usuario)

            usuario = {
                'id': id_usuario,
                'nombre': nombre,
                'apellido': apellido,
                'telefono': telefono,
                'correo': correo
            }

            total_usuarios.append(usuario)
        	

        print('Cantidad de usuarios registrados: ' + str(cantidad_usuarios))
        print(' ')

        #print('Identificadores: ' + str(identificadores))

        #print(usuario)

    if opcion == 2:     # Lista los identificadores de usuarios registrados
        print('Identificadores: ' + str(identificadores))
        print(' ')

    if opcion == 3:     #Lista de datos del usuario segun su ID
        identificador = int(input('Ingrese el ID del usuario a buscar: '))
        for _usuario in total_usuarios:
            if _usuario['id'] == identificador:
                print('Nombre completo: '+ _usuario['nombre'] + ' ' + _usuario['apellido'])
                print('Telefono: '+ _usuario['telefono'])
                print('Correo: '+ _usuario['correo'])
    
    if opcion == 4:     #Editar datos del usuario segun su ID
        identificador = int(input('Ingrese el ID del usuario a modificar: '))
        print('Datos actuales del usuario:')
        for _usuario in total_usuarios:
            if _usuario['id'] == identificador:
                print('Nombre completo: '+ _usuario['nombre'] + ' ' + _usuario['apellido'])
                print('Telefono: '+ _usuario['telefono'])
                print('Correo: '+ _usuario['correo'])
        
        print(' ')

        nombre = str(input('Ingrese el nuevo nombre: '))
        apellido = str(input('Ingrese el nuevo apellido: '))
        telefono = input('Ingrese el nuevo telefono: ')
        correo = str(input('Ingrese el nuevo correo: '))

        for _usuario in total_usuarios:
            if _usuario['id'] == identificador:
                _usuario['nombre'] = nombre
                _usuario['apellido'] = apellido
                _usuario['telefono'] = telefono
                _usuario['correo'] = correo
        
        print(' ')

        print('Datos actualizados:')
        print(' ')

        for _usuario in total_usuarios:
            if _usuario['id'] == identificador:
                print('Nombre completo: '+ _usuario['nombre'] + ' ' + _usuario['apellido'])
                print('Telefono: '+ _usuario['telefono'])
                print('Correo: '+ _usuario['correo'])  

        print(' ')

    print(' ')  
else:
    print('Finalizado')