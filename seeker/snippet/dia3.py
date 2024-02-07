#date: 2024-02-07T17:01:32Z
#url: https://api.github.com/gists/4b2ba76e9d2bd2b4fd9ce4960d69f41e
#owner: https://api.github.com/users/ChristYela

def validar_longitud(valor, minimo, maximo):
    return minimo <= len(valor) <= maximo

def validar_telefono(telefono):
    return len(telefono) == 10 and telefono.isdigit()

def registrar_usuarios():

    cantidad_usuarios = int(input("Ingrese la cantidad de nuevos usuarios a registrar: "))
    
    ids_usuarios = []

    for i in range(1, cantidad_usuarios + 1):

        nombres = input("Ingrese su(s) nombre(s): ")
        apellidos = input("Ingrese sus apellidos: ")
        telefono = input("Ingrese su número de teléfono: ")
        correo = input("Ingrese su correo electrónico: ")

        if not (validar_longitud(nombres, 5, 50) and validar_longitud(apellidos, 5, 50) and validar_longitud(correo, 5, 50)):
            print("Error: Los nombres, apellidos y correo deben tener entre 5 y 50 caracteres.")
            return

        if not validar_telefono(telefono):
            print("Error: El número de teléfono debe tener 10 dígitos.")
            return

        id_usuario = i
 
        ids_usuarios.append(id_usuario)

        print(f"Registro exitoso - ID: {id_usuario} - Hola {nombres} {apellidos}, en breve recibirás un correo a {correo}.")

    print("\nIDs de usuarios registrados:")
    for id_usuario in ids_usuarios:
        print(id_usuario)

registrar_usuarios()

