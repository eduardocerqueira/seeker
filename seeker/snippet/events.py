#date: 2023-07-28T16:42:03Z
#url: https://api.github.com/gists/3e4d11c2986c930521ec991a234e7c63
#owner: https://api.github.com/users/JuanCarlosMarino

# eventos = [{"name": "paseo de rio", "location": "rio negro", "day": 15, "is done": False},
#            {"name": "paseo de rio", "location": "rio negro", "day": 15, "is done": False},
#            {"name": "paseo de rio", "location": "rio negro", "day": 15, "is done": False}
#         ]

events = [{'name': 'Misa', 'location': 'Iglesia 123', 'day': 15, 'is done': True}, 
          {'name': 'paseo de rio', 'location': 'rio negro', 'day': 20, 'is done': False}
        ]

def create():
    print("**********************************************************************")
    eve = {}
    name = input("Ingrese el nombre del evento: ")
    location = input("Ingrese la ubicación del evento: ")
    day = int(input("Ingrese el día del evento: "))
    eve = {"name": name, "location": location, "day": day, "is done": False}
    events.append(eve)
    print("Evento creado!!")
    print("**********************************************************************")

def update():
    print("**********************************************************************")
    print("Ingrese el número del evento a modificar según la lista: ")
    for i in range(len(events)):
        print(i, "-", events[i])
    opc = int(input("Ingrese la opción deseada: "))
    if opc >= 0 and opc < len(events):
        if events[opc]["is done"]:
            print("Evento ya ejecutado!!")
        else:
            name = input("Ingrese el nombre del evento: ")
            location = input("Ingrese la ubicación del evento: ")
            day = int(input("Ingrese el día del evento: "))
            eve = {"name": name, "location": location, "day": day, "is done": False}
            events[opc] = eve
            print("Evento actualizado!!")
            print(events)
    else:
        print("Opción no valida")
    print("**********************************************************************")
    
def delete():
    print("**********************************************************************")
    print("Ingrese el número del evento a eliminar según la lista: ")
    for i in range(len(events)):
        print(i, "-", events[i])
    opc = int(input("Ingrese la opción deseada: "))
    if opc >= 0 and opc < len(events):
        if events[opc]["is done"]:
            print("Evento ya ejecutado!!")
        else:
            events.pop(opc)
            print("Evento eliminado!!")
            print(events)
    else:
        print("Opción no valida")
    print("**********************************************************************")
    
def change_is_done():
    print("**********************************************************************")
    print("Ingrese el número del evento a marcar como realizado: ")
    for i in range(len(events)):
        print(i, "-", events[i])
    opc = int(input("Ingrese la opción deseada: "))
    if opc >= 0 and opc < len(events):
        if events[opc]["is done"]:
            print("Evento ya ejecutado!!")
        else:
            events[opc]["is done"] = True
            print("Evento marcado como ejecutado!!")
            print(events)
    else:
        print("Opción no valida")
    print("**********************************************************************")