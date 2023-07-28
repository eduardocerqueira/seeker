#date: 2023-07-28T16:42:03Z
#url: https://api.github.com/gists/3e4d11c2986c930521ec991a234e7c63
#owner: https://api.github.com/users/JuanCarlosMarino

# participantes = {
#     "1234567890":{"name":"Juan Mariño", "age": 55, "area": "operations", "is paid": False},
#     "1234567890":{"name":"Juan Mariño", "age": 55, "area": "operations", "is paid": False},
#     "1234567890":{"name":"Juan Mariño", "age": 55, "area": "operations", "is paid": False}
# }

users = {}

def create():
    while True:
        print("*****************************************************************")
        print("Creación del participante")
        document = input("Ingrese el documento del participante: ")
        name = input("Ingrese el nombre del participante: ")
        age = input("Ingrese la edad: ")
        area = input("Ingrese el area a la que pertenece el participante: ")
        option = input("Ingrese s si pagó, o ingrese n si no pagó: ")
        is_paid = False
        print("*****************************************************************")
        if option == "s" or option == "S":
            is_paid =  True
        if document != " ":
            users[document] = {"name": name, "age": age, "area":area, "is paid": is_paid}
            break

def remove_user():
    print("*****************************************************************")
    document = input("Ingrese el documento a eliminar: ")
    user = users.get(document, False)
    if user == False:
        print("Usuario no registrado :(")
    elif user["is paid"]:
        print("Ya pagaste, ya perdiste")
    else:
        del users[document]
        print("Usuario eliminado!")    
    print("*****************************************************************")
    
def pay_register():
    print("*****************************************************************")
    document = input("Ingrese el documento del participante a pagar: ")
    user = users.get(document, False)
    if user == False:
        print("Usuario no registrado :(")
    elif user["is paid"]:
        print("Ya pagaste, no tienes que pagar")
    else:
        users[document]["is paid"] = True
        print("Pagaste!")    
    print("*****************************************************************")    