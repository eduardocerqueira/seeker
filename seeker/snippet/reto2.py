#date: 2024-02-06T16:55:45Z
#url: https://api.github.com/gists/e4f861b62ebe491f8f46de27fc44fb2b
#owner: https://api.github.com/users/Anitandil

#RETO 2 - CODIGO FACILITO - Registrar n usuarios
print("* REGISTRO DE NUEVOS USUARIOS *")
cantidad = int(input("Cuantos usuarios desea registrar?: "))

for i in range(cantidad):
    print()
    print(F"USUARIO N° {i + 1}")
    nombre = input("Ingrese el nombre completo: ")
    while len(nombre) < 5 or len(nombre) >50:
        print("ERROR: el nombre debe contener entre 5 y 50 caracteres!")
        nombre = input("Ingrese el nombre completo: ")
    apellido= input("Ingrese el apellido: ")
    while len(apellido) < 5 or len(apellido) >50:
        print("ERROR: el apellido debe contener entre 5 y 50 caracteres!")
        apellido= input("Ingrese el apellido: ")
    telefono = input("Ingrese el numero telefonico: ")
    while len(str(telefono)) !=10 or not telefono.isdigit():
        print("ERROR: el telefono debe contener 10 digitos!")
        telefono = input("Ingrese el numero telefonico: ")
    mail = input("Ingrese el email: ")
    while len(mail) < 5 or len(mail) >50:
        print("el email debe contener entre 5 y 50 caracteres!")
        mail = input("Ingrese el email: ")
        
    print()
    print("NUEVO USUARIO CREADO")
    print(f"Nombre: {nombre} {apellido}")
    print(f"Teléfono: {telefono} ")
    print(f"Email: {mail} ")