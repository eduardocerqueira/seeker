#date: 2024-02-08T16:57:18Z
#url: https://api.github.com/gists/a7b4510422e5347aa029db835a4dbf9a
#owner: https://api.github.com/users/rudolfwolf

# Inicialización de variables
next_id = 0
usuarios = {}  # Diccionario para almacenar los usuarios

# Función para validar la entrada del usuario
def validacion(mensaje, maximo, minimo):
    while True:
        entrada = input(mensaje)
        if minimo <= len(entrada) <= maximo:
            return entrada
        else:
            print(f"Error: La entrada debe tener entre {minimo} y {maximo} caracteres. Intenta nuevamente.")

# Función para agregar un nuevo usuario
def agregar_usuario():
    global next_id
    next_id += 1
    # Se solicita al usuario que ingrese sus datos
    name = validacion('Ingrese su(s) nombre(s) por favor: ', 50, 5)
    lastname = validacion('Ingrese su(s) apellido(s) por favor: ', 50, 5)
    email = validacion('Ingrese su e-mail por favor: ', 50, 5)
    tel = validacion('Ingrese su teléfono a 10 dígitos por favor: ', 10, 10)
    # Se agrega el usuario al diccionario de usuarios
    usuarios[next_id] = {'nombre': name, 'apellido': lastname, 'email': email, 'telefono': tel}
    print(f'Hola {name} {lastname}, en breve recibirás un correo a {email}\n')
    print(f'Se ha registrado el usuario con ID: {next_id}')

# Función para listar todos los IDs de usuarios registrados
def listar_usuarios():
    print("Lista de IDs de usuarios registrados:")
    for user_id in usuarios:
        print(f"ID: {user_id}")

# Función para consultar la información de un usuario por su ID
def consultar_usuario():
    user_id = int(input("Ingrese el ID del usuario que desea consultar: "))
    if user_id in usuarios:
        usuario = usuarios[user_id]
        print(f"Información del usuario con ID {user_id}:")
        print(f"Nombre: {usuario['nombre']}")
        print(f"Apellido: {usuario['apellido']}")
        print(f"Email: {usuario['email']}")
        print(f"Teléfono: {usuario['telefono']}")
    else:
        print("No se encontró ningún usuario con ese ID.")

# Función para editar la información de un usuario por su ID
def editar_usuario():
    user_id = int(input("Ingrese el ID del usuario que desea editar: "))
    if user_id in usuarios:
        print(f"Editando información del usuario con ID {user_id}:")
        # Se solicita al usuario que ingrese los nuevos datos
        name = validacion('Ingrese su(s) nombre(s) por favor: ', 50, 5)
        lastname = validacion('Ingrese su(s) apellido(s) por favor: ', 50, 5)
        email = validacion('Ingrese su e-mail por favor: ', 50, 5)
        tel = validacion('Ingrese su teléfono a 10 dígitos por favor: ', 10, 10)
        # Se actualiza la información del usuario en el diccionario
        usuarios[user_id] = {'nombre': name, 'apellido': lastname, 'email': email, 'telefono': tel}
        print(f"Se ha actualizado la información del usuario con ID {user_id}")
    else:
        print("No se encontró ningún usuario con ese ID.")

# Menú principal
while True:
    print("\n--- [Menú] ---\n")
    print("[1] Agregar usuario")
    print("[2] Listar usuarios")
    print("[3] Consultar usuario por ID")
    print("[4] Editar usuario por ID")
    print("[5] Salir")

    opcion = input("\nSeleccione una opción: ")
    if opcion == '1':
        agregar_usuario()
    elif opcion == '2':
        listar_usuarios()
    elif opcion == '3':
        consultar_usuario()
    elif opcion == '4':
        editar_usuario()
    elif opcion == '5':
        print("Saliendo...")
        break
    else:
        print("Opción inválida. Por favor, seleccione una opción válida.")
