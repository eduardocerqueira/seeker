#date: 2024-02-09T17:08:57Z
#url: https://api.github.com/gists/fd12d99e002420f7ba7eafa1f3ca56cb
#owner: https://api.github.com/users/Ejjq

# Codigo para el reto del dia jueves.
# Eddie José Juárez
print()
print('----------Utilidad de Regsitro de Usuarios-----------')
print()
datos_usuarios={} #inicialización del diccionario
while True:
    print()
    print('              Menu Principal            ')
    print('Opciones disponibles para el manejo de datos de Usuarios')
    print(
        '1. Ingreso de Usuarios \n'
        '2. Mostrar listado de ID de todos los usuarios\n'
        '3. Verificar la información de usuario por su ID\n'
        '4. Editar la información por medio de un ID determinado\n'
        '5. Salir del programa \n'
    )
    
    
    seleccion=input('Que numero de opción quiere realizar (1-5): ')
    if seleccion == '1':
        
        total_usuarios=int(input('Indique el numero de usuarios a ingresar: '))
        registro=100 #El identificador ID
            
        for total in range(1,total_usuarios+1):
            listas_usuarios=[]
            print('No.->',total) #Correlativo de la cantidad de usuarios a ingresar.
            while True: #Ingreso de los nombres de usuario
                nombres=input('Ingrese sus nombres: ')
                cont1=0
                for nom in nombres:
                    cont1+=1
                if cont1 >=5 and cont1<=50: #validación de que los nombres esten entre 5 y 50 caracteres.
                    listas_usuarios.append(nombres)
                    break
                else:
                        print('Nombres incorrectos...Ingrese de nuevo')
                
                
            while True: #ingreso de los apellidos de usuario

                apellidos=input('Ingrese sus apellidos: ')
                cont2=0
                for apellido in apellidos:
                    cont2+=1
                if cont2 >=5 and cont1<=50: #validación de que los apellidos esten entre 5 y 50 caracteres.
                    listas_usuarios.append(apellidos)
                    break
                else:
                    print('Apellidos incorrectos...Ingrese de nuevo')
                

            while True: #Ingreso del numero telefonico de usuario

                num_telefono=input('Ingrese su numero telefonico: ')
                cont3=0
                for num in num_telefono:
                    cont3+=1
                if cont3 == 10: #validación de que el numero telefonico tenga 10 digitos
                    listas_usuarios.append(num_telefono)
                    break
                else:
                    print('Numero Telefonico incorrecto...Ingrese de nuevo')
                

            while True: #Ingreso del correo electronico de usuario

                correo_electronico=input('Ingrese su correo electronico: ')
                cont4=0
                for correo in correo_electronico:
                    cont4+=1
                if cont4 >=5 and cont1<=50: #validación de que el correo electronico este entre 5 y 50 caracteres
                    listas_usuarios.append(correo_electronico)
                    break
                else:
                    print('correo electronico incorrecto...Ingrese de nuevo')
                
                
            print('Hola'+' '+ nombres+' '+ apellidos+ '.'' En breve recibiras un correo a: '+ correo_electronico)
            registro+=1
            datos_usuarios[registro]=listas_usuarios
        print()
        input('Enter para continuar...')
                


    elif seleccion =='2': # codigo para mostrar los registros ingresados
        print('Los ID´s de los usuarios en la base de datos son:' )
        usuarios=datos_usuarios.keys()
        for llave in usuarios:
            print()
            print('El ID de usuario es: ', llave, 'Y su registro es: ', datos_usuarios[llave])
        print()
        input('Enter para continuar...')

    elif seleccion=='3': # codigo para verificar la informacion de un usuario seleccionado por su ID
            id_usuario=int(input('Si desea ver la información de un usuario determinado, ingrese su numero de ID--> '))
            dato_usuario=datos_usuarios[id_usuario]
            nombres, apellidos, num_telefono, email = dato_usuario 
            print('los nombres del usuario son: ', nombres)
            print('los apellidos del usuario son:', apellidos)
            print('El numero de telefono del usuario es: ', num_telefono)
            print('El correo electronico del usuario es: ', email)
            print()
            input('Enter para continuar....')
    
    elif seleccion=='4': # codigo para editar la informacion de un usuario por medio de su ID
            id_usuario=int(input('Si desea modificar la información de un usuario determinado, ingrese su numero de ID--> '))
            dato_usuario=datos_usuarios[id_usuario]
            while True:
                nuevo_nombre=input('Ingrese los nuevos nombres del usuario ')
                if len(nuevo_nombre)>= 5 and len(nuevo_nombre)<=50:
                     break
                else:
                     print('Valores incorrectos, Vuelva a intentarlo ')
            while True:
                nuevo_apellidos=input('Ingrese los nuevos apellidos del usuario ')
                if len(nuevo_apellidos)>= 5 and len(nuevo_apellidos)<=50:
                     break
                else:
                     print('Valores incorrectos, Vuelva a intentarlo ')
            while True:
                nuevo_tel=(input('Ingrese el nuevo numero de telefono '))
                if len(nuevo_tel) == 10:
                     break
                else:
                     print('Valores incorrectos, Vuelva a intentarlo ')
            while True:
                 nuevo_email=input('Ingrese el nuevo correo electronico: ')
                 if len(nuevo_email) >= 5 and len(nuevo_email) <=50:
                      break
                 else:
                      print('Valores incorrectos, Vuelva a intentarlo')
            print()
            dato_usuario[0]=nuevo_nombre
            dato_usuario[1]=nuevo_apellidos
            dato_usuario[2]=nuevo_tel
            dato_usuario[3]=nuevo_email
            datos_usuarios[id_usuario]=dato_usuario
            input('Enter para continuar....')
    elif seleccion=='5': # salir del programa
            print()
            
            
            print('-----------------Hasta la proxima..!----------')
            break
    else:
            print('<<<<<<<<<<<<La opción no esta disponible. Vuelva a intentarlo>>>>>>>>>>>')