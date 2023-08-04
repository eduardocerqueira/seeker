#date: 2023-08-04T16:56:08Z
#url: https://api.github.com/gists/d56e95176b697d5fdeb9a3ad6cd16266
#owner: https://api.github.com/users/AitsuYuyu

import workersDB
from datetime import datetime

def register():
    workers = workersDB.cargar('workers.json')
    while True:
        print('Bienvenido:)')
        name = input('Ingrese el nombre del trabajador\n')
        dni = input(f"Identificación de: {name}: ")
        if workers.get(dni, False) != False:
            print('ya estas sapo')
            continue        
        try:
            phone_number=int(input('Ingrese el numero de telefono de '+name+": "))
        except ValueError:
            print('invalid input')
            continue
        else: 
            workers[dni] = {'name': name, 'phone':phone_number, 'isActive': True}
            break
    workersDB.guardar('workers.json', workers)
            
def listar():
    workers=workersDB.cargar("workers.json")
    for i in workers.keys():
        if workers[i]["isActive"]:
            print("El empleado ", workers[i]["name"]," de indentificacion ",i," de telefono ",workers[i]["phone"])
        
def modificarEmpleado():
    workers=workersDB.cargar("workers.json")
    dni = input("Ingrese el documento que desea consultar: ")
    if workers.get(dni, False) != False:
        name = input("Ingresa el nombre modificado: ")
        try:
            phone_number=int(input("Ingrese el numero de telefono: ")) 
        except ValueError:
            print('invalid input')            
        else: 
            workers[dni] = {'name': name, 'phone':phone_number, 'isActive':  workers[dni]['isActive'] }
            print("Empleado modificado")
            
        workersDB.guardar('workers.json', workers)

def despedir_empleado():
    workers=workersDB.cargar("workers.json")
    dni = input("Ingrese el documento del empleado a despedir: ")
    if workers.get(dni, False) != False:
        workers[dni]['isActive']=False
            
        workersDB.guardar('workers.json', workers)
        print("Empleado Despedido")

def in_out():
    workers=workersDB.cargar("workers.json")
    linea = []
    dni = input("Ingrese el documento del empleado a registrar: ")
    if workers.get(dni, False) != False:
        linea.append(dni)
        print("Ingrese\n1. para registrar entrada\n2. para registrar salida")
        try:
            option=int(input("Ingrese la opción: "))
            advertencia = ""
            if option == 1 or option ==2:
                if option == 1:
                    linea.append("in")
                else:
                    linea.append("out")
                    
                time = datetime.now()
                moment = time.strftime("%d-%m-%Y %H:%M")
                linea.append(moment)
                
                
                hour = time.hour
                minute = time.minute
                if option == 1:
                    if hour > 8:
                        advertencia = "LLegó tarde"
                    elif hour == 8 and minute > 0:
                        advertencia = "LLegó tarde"
                else:
                    if hour < 18:
                        advertencia = "Se fue temprano"
                
                if advertencia != "":
                    linea.append(advertencia)
                
                workersDB.guardar_registro("registros.csv", linea)
            else:
                print("Opción no válida")
        except ValueError:
            print('invalid input')            
    else:
        print("Empleado no existente!") 