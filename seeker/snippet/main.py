#date: 2024-02-08T16:58:09Z
#url: https://api.github.com/gists/498964ad20c245098454f0d2e0db1711
#owner: https://api.github.com/users/Velaeli

cant_Usuarios = int(input("Cuantos usuarios desea añadir: "))
i = 0
id_list = []
id = 0

for i in range(cant_Usuarios):
    while True:
        name = input("Ingrese sus nombres: ")
        if len(name) < 5:
            print("El nombre es corto. Ingrese un nombre entre 5 y 50 caracteres")
        elif len(name) > 50:
            print("El nombre es muy largo. Ingrese un nombre entre 5 y 50 caracteres")
        else:
            print("Nombre guardado")
            break

    while True:
        surnames = input("Digite sus apellidos: ")
        if len(surnames) < 5:
            print("El apellido es corto. Ingrese un nombre entre 5 y 50 caracteres")
        elif len(surnames) > 50:
            print("El apellido es muy largo. Ingrese un nombre entre 5 y 50 caracteres")
        else:
            print("Apellido guardado")
            break

    while True:
        phone = input("Digite su numero telefonico: ")
        if len(phone) < 10:
            print("El numero telefonico es corto. Debe tener 10 caracteres")
        elif len(phone) > 10:
            print("El numero telefonico es muy largo. Debe tener 10 caracteres")
        else:
            print("Numero telefonico guardado")
            break

    while True:
        email = input("Ingresar su correo electronico: ")
        if len(email) < 5:
            print("El email es corto. Ingrese un nombre entre 5 y 50 caracteres")
        elif len(email) > 50:
            print("El email es muy largo. Ingrese un nombre entre 5 y 50 caracteres")
        else:
            print("Email guardado")
            break

    id += 1
    id_list.append(id)

    print(
        "Hola "
        + name
        + " "
        + surnames
        + " "
        + "en breve recibirás un correo a "
        + email
        + " Su identificador es"
        + " "
        + str(id)
    )
    i += 1

print("Haz ingresado todos los usuarios, la lista de los ID registrado es:", id_list)