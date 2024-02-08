#date: 2024-02-08T16:53:35Z
#url: https://api.github.com/gists/f848942401ad1650398dca3d412bac05
#owner: https://api.github.com/users/Derssavl1405


nombre = input("Ingrese su nombre: ")
    # Validación de nombre
while not (5 <= len(nombre) <= 50):
        print("El nombre debe tener entre 5 y 50 caracteres.")
        nombre = input("Ingrese su nombre: ")

apellidos = input("Ingrese sus apellidos: ")
    # Validación de apellidos
while not (5 <= len(apellidos) <= 50):
        print("Los apellidos deben tener entre 5 y 50 caracteres.")
        apellidos = input("Ingrese sus apellidos: ")

telefono = input("Ingrese su número de teléfono: ")
    # Validación de teléfono
while not (len(telefono) == 10 and telefono.isdigit()):
        print("El número de teléfono debe tener exactamente 10 dígitos.")
        telefono = input("Ingrese su número de teléfono: ")

correo = input("Ingrese su correo electrónico: ")
    # Validación de correo electrónico
while not (5 <= len(correo) <= 50 and '@' in correo and '.' in correo):
        print("El correo electrónico debe tener entre 5 y 50 caracteres y ser válido.")
        correo = input("Ingrese su correo electrónico: ")

   

# Función principal
cantidad_usuarios = int(input("¿Cuántos nuevos usuarios desea registrar? "))
while cantidad_usuarios <= 0:
    print("La cantidad de usuarios debe ser un número positivo mayor que cero.")
    cantidad_usuarios = int(input("¿Cuántos nuevos usuarios desea registrar? "))

usuarios = []
for i in range(cantidad_usuarios):
    print(f"\nIngrese los datos del usuario {i+1}:")
    nombre = input("Ingrese su nombre: ")
    # Validación de nombre
    while not (5 <= len(nombre) <= 50):
        print("El nombre debe tener entre 5 y 50 caracteres.")
        nombre = input("Ingrese su nombre: ")

    apellidos = input("Ingrese sus apellidos: ")
    # Validación de apellidos
    while not (5 <= len(apellidos) <= 50):
        print("Los apellidos deben tener entre 5 y 50 caracteres.")
        apellidos = input("Ingrese sus apellidos: ")

    telefono = input("Ingrese su número de teléfono: ")
    # Validación de teléfono
    while not (len(telefono) == 10 and telefono.isdigit()):
        print("El número de teléfono debe tener exactamente 10 dígitos.")
        telefono = input("Ingrese su número de teléfono: ")

    correo = input("Ingrese su correo electrónico: ")
    # Validación de correo electrónico
    while not (5 <= len(correo) <= 50 and '@' in correo and '.' in correo):
        print("El correo electrónico debe tener entre 5 y 50 caracteres y ser válido.")
        correo = input("Ingrese su correo electrónico: ")

    usuarios.append((nombre, apellidos, telefono, correo))

print("\nLos nuevos usuarios han sido registrados exitosamente:")
for i, usuario in enumerate(usuarios, start=1):
    print(f"\nUsuario {i}:")
    print(f"Nombre: {usuario[0]}")
    print(f"Apellidos: {usuario[1]}")
    print(f"Teléfono: {usuario[2]}")
    print(f"Correo electrónico: {usuario[3]}")