#date: 2024-02-07T16:51:29Z
#url: https://api.github.com/gists/ebe8869548acebb707cb459d1c5e2254
#owner: https://api.github.com/users/AngelP1105

cantidad_usuarios_ingresar = int(input("Cuantos usuarios se van a ingresar? "))
listas_id = []

for i in range(cantidad_usuarios_ingresar):
    error = True
    print("Bienvenido nuevo usuario")
    while error:
        fallo = ""
        nombres_usuario = input("Inserte sus nombres: ")
        longitud_nombre = len(nombres_usuario)
        if longitud_nombre < 5 or longitud_nombre > 50:
            fallo = "Nombre"
        apellidos_usuario = input("Inserte sus apellidos: ")
        longitud_apellido = len(apellidos_usuario)
        if longitud_apellido < 5 or longitud_apellido > 50:
            fallo = "Apellido"
        numero_usuario = input("Inserte su numero de telefono: ")
        longitud_telefono = len(numero_usuario)
        if longitud_telefono != 10:
            fallo = "Numero"
        correo_usuario = input("inserte su correo electronico: ")
        longitud_correo = len(correo_usuario)
        if longitud_correo < 5 or longitud_correo > 50:
            fallo = "Correo"

        match fallo:
            case "Nombre":
                print("El nombre debe tener una longitud minima de 5 y maxima de 50")
            case "Apellido":
                print("El apellido debe tener una longitud minima de 5 y maxima de 50")
            case "Correo":
                print("El correo debe tener una longitud minima de 5 y maxima de 50")
            case "Numero":
                print("El numero debe tener 10 digitos")
            case _:
                error = False
                listas_id.append(i)

print("Los usuarios se han registrado con exito")
print(listas_id)