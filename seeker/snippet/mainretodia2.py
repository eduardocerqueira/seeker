#date: 2024-02-09T16:42:28Z
#url: https://api.github.com/gists/6b69bae6d41fe01801cf20d974a5a303
#owner: https://api.github.com/users/rmontesch

# Reto dia 2
salir = 'n'
records = 0
#random, number, num_records = 5, 0, 1
while salir != 's':
    num_records = int(input('cuantos registros deseas ingresar '))
    while records < num_records:
        names = input('Ingresa tu nombre(s) : ')
        if len(names) < 5 or len(names) > 50 :
            print('El nombre debe de ser de longitud entre 5 y 50 caracteres)')
        else:
            last_name = input('ingresa tus apellidos :')
            if len(last_name) < 5 or len(last_name) > 50 :
                print('Apellido(s) debe de ser de longitud entre 5 y 50 caracteres)')
            else:
                full_name = names + ' ' + last_name
                #telephone_number = int(input('Ingresa tu número telefónico :'))
                telephone_number = (input('Ingresa tu número telefónico :'))
                if len(telephone_number) < 10 :
                    print('Telefono debe de tener 10 digitos')
                else:
                    telephone_number = int(telephone_number)
                    email = input('Ingresa tu correo electrónico :')
                    if len(email) < 5 or len(email) > 50 :
                        print('Correo electrónico debe de ser de longitud entre 5 y 50 caracteres)')
                    else:    
                        print ('Hola ' + full_name +', en breve recibiras un correo a '+ email)
                        #Aqui se debe adicionar el registro y pasamos al siguiente
                        records += 1
    else:
        print('Haz registrado los '+ str(num_records)+' que solicitaste')
        salir ='s'
        #exit
#else:
#    print('Necesitas un número mayor a 0 para iniciar') 