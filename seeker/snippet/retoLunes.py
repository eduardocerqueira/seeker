#date: 2024-02-05T17:10:09Z
#url: https://api.github.com/gists/0dc59c307d532ba612e2afa729c5dfbd
#owner: https://api.github.com/users/EnriqueGT27

#Reto: Programa que permita crear a un usuario
#Nombres, apellidos, numero de telefono, correo
#Una vez ingresados sus datos el programa deberá de darle la bienvenida 
#Hola NOMBRE, en breve recibirás un correo a CORREO

print()
print('-- Alta de usuario --')
print()

user_name = str(input('Ingresa tu nombre: '))
user_last_name = str(input('Ingresa tus apellidos: '))

user_phone_number = str(input('Ingresa tu numbero telefonico: '))

user_email = str(input('Ingresa tu dirección de correo electronico: '))

print()
print('Cargando tu información . . .')
print()

print('Éxito.')
print()

print('Hola ' + user_name + ' ' + user_last_name + ', ' + 'en breve recibirás un correo en la dirección: ' + user_email)
print()