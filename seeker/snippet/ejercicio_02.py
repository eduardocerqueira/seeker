#date: 2024-02-06T17:04:56Z
#url: https://api.github.com/gists/7210e72bc59f61d9b3eb5f8b69b2dff8
#owner: https://api.github.com/users/DannaGuiot

def pedir_usuario():
    nombre = input("Ingrese su(s) nombre(s): ")
    apellidos = input("Ingrese sus apellidos: ")
    telefono = input("Ingrese su número de teléfono (10 dígitos): ")
    correo = input("Ingrese su correo electrónico: ")

    if len(nombre) < 5 or len(nombre) > 50:
        print("El nombre debe tener entre 5 y 50 caracteres.")
        return False
    if len(apellidos) < 5 or len(apellidos) > 50:
        print("Los apellidos deben tener entre 5 y 50 caracteres.")
        return False
    if len(correo) < 5 or len(correo) > 50:
        print("El correo electrónico debe tener entre 5 y 50 caracteres.")
        return False

    if len(telefono) != 10 or not telefono.isdigit():
        print("El número de teléfono debe tener exactamente 10 dígitos.")
        return False

    return nombre, apellidos, telefono, correo


def main():
    num_usuarios = int(input("¿Cuántos usuarios desea registrar? "))
    usuarios = []

    for i in range(num_usuarios):
        print(f"\nUsuario {i + 1}:")
        usuario_valido = False
        while not usuario_valido:
            usuario = pedir_usuario()
            if usuario:
                usuarios.append(usuario)
                usuario_valido = True

    print("\nUsuarios registrados correctamente:")
    for usuario in usuarios:
        print("Nombre:", usuario[0])
        print("Apellidos:", usuario[1])
        print("Teléfono:", usuario[2])
        print("Correo electrónico:", usuario[3])
        print()


if __name__ == "__main__":
    main()
    

