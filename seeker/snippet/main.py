#date: 2024-02-13T16:46:06Z
#url: https://api.github.com/gists/f1e1b07b736bd5897debc55d9eafed50
#owner: https://api.github.com/users/MartinAAcebeyL

def validar_nombre(nombre):
    return 5 <= len(nombre) <= 50

def validar_telefono(telefono):
    return len(telefono) == 10 and telefono.isdigit()

def validar_correo(correo):
    return 5 <= len(correo) <= 50 and '@' in correo and '.' in correo

def new_user(usuarios_registrados):
    nombre = input("Ingrese su nombre(s): ")
    while not validar_nombre(nombre):
        print("El nombre debe tener entre 5 y 50 caracteres.")
        nombre = input("Ingrese su nombre(s): ")

    apellidos = input("Ingrese su apellido(s): ")
    while not validar_nombre(apellidos):
        print("Los apellidos deben tener entre 5 y 50 caracteres.")
        apellidos = input("Ingrese su apellido(s): ")

    telefono = input("Ingrese su número de teléfono: ")
    while not validar_telefono(telefono):
        print("El número de teléfono debe tener 10 dígitos.")
        telefono = input("Ingrese su número de teléfono: ")

    correo = input("Ingrese su correo electrónico: ")
    while not validar_correo(correo):
        print("El correo electrónico debe tener entre 5 y 50 caracteres y ser válido.")
        correo = input("Ingrese su correo electrónico: ")

    id_usuario = len(usuarios_registrados) + 1
    usuario = {
        "id": id_usuario,
        "nombre": nombre,
        "apellidos": apellidos,
        "telefono": telefono,
        "correo": correo
    }
    usuarios_registrados.append(usuario)
    print("Usuario registrado exitosamente con ID:", id_usuario)

def show_user(usuarios_registrados, id_usuario):
    for usuario in usuarios_registrados:
        if usuario["id"] == id_usuario:
            print("\nInformación del usuario con ID", id_usuario)
            print("Nombre:", usuario["nombre"], usuario["apellidos"])
            print("Teléfono:", usuario["telefono"])
            print("Correo:", usuario["correo"])
            return
    print("No se encontró ningún usuario con el ID proporcionado.")

def edit_user(usuarios_registrados, id_usuario):
    for usuario in usuarios_registrados:
        if usuario["id"] == id_usuario:
            print("\nEditando información del usuario con ID", id_usuario)
            nombre = input("Ingrese el nuevo nombre(s): ")
            while not validar_nombre(nombre):
                print("El nombre debe tener entre 5 y 50 caracteres.")
                nombre = input("Ingrese el nuevo nombre(s): ")

            apellidos = input("Ingrese los nuevos apellido(s): ")
            while not validar_nombre(apellidos):
                print("Los apellidos deben tener entre 5 y 50 caracteres.")
                apellidos = input("Ingrese los nuevos apellido(s): ")

            telefono = input("Ingrese el nuevo número de teléfono: ")
            while not validar_telefono(telefono):
                print("El número de teléfono debe tener 10 dígitos.")
                telefono = input("Ingrese el nuevo número de teléfono: ")

            correo = input("Ingrese el nuevo correo electrónico: ")
            while not validar_correo(correo):
                print("El correo electrónico debe tener entre 5 y 50 caracteres y ser válido.")
                correo = input("Ingrese el nuevo correo electrónico: ")

            usuario["nombre"] = nombre
            usuario["apellidos"] = apellidos
            usuario["telefono"] = telefono
            usuario["correo"] = correo
            print("Información actualizada correctamente.")
            return
    print("No se encontró ningún usuario con el ID proporcionado.")

def delete_user(usuarios_registrados, id_usuario):
    for usuario in usuarios_registrados:
        if usuario["id"] == id_usuario:
            usuarios_registrados.remove(usuario)
            print("Usuario con ID {} eliminado correctamente.".format(id_usuario))
            return
    print("No se encontró ningún usuario con el ID proporcionado.")

def list_users(usuarios_registrados):
    print("\nUsuarios registrados:")
    for usuario in usuarios_registrados:
        print("ID:", usuario["id"], "- Nombre:", usuario["nombre"], usuario["apellidos"], "- Teléfono:", usuario["telefono"], "- Correo:", usuario["correo"])

def menu():
    print("\nMenú de opciones:")
    print("A. Registrar nuevo usuario")
    print("B. Listar usuarios")
    print("C. Ver información de un usuario")
    print("D. Editar información de un usuario")
    print("E. Eliminar usuario")
    print("F. Salir")

usuarios_registrados = []

options = {
    "A": new_user,
    "B": list_users,
    "C": show_user,
    "D": edit_user,
    "E": delete_user,
}

while True:
    menu()
    opcion = input("Ingrese la opción deseada: ").upper()

    if opcion == "F":
        print("¡Hasta luego!")
        break
    elif opcion in options:
        if opcion == "B":
            options[opcion](usuarios_registrados)
        else:
            try:
                id_usuario = int(input("Ingrese el ID del usuario: "))
                options[opcion](usuarios_registrados, id_usuario)
            except ValueError:
                print("ID inválido. Por favor, ingrese un ID numérico válido.")
    else:
        print("Opción inválida. Por favor, seleccione una opción válida del menú.")
