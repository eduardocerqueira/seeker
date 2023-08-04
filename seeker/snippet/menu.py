#date: 2023-08-04T16:56:08Z
#url: https://api.github.com/gists/d56e95176b697d5fdeb9a3ad6cd16266
#owner: https://api.github.com/users/AitsuYuyu

import workers

def menuPrincipal():
    while True:
        print("---------------------------")
        print("         MENU PRINCIPAL    ")
        print("---------------------------")
        
        print("1. Registrar empleado")
        print("2. Listar empleados")
        print("3. Modificar empleados")
        print("4. Despedir empleados")
        print("5. Registrar entrada o salida")
        print("0. Salir")
        
        try:
            option = int(input("Digite la opcion a realizar: "))
        except ValueError:
            print("Ingrese un valor numerico")
        else:
            if option == 1:
                workers.register()
            elif option == 2:
                workers.listar()
            elif option == 3:
                workers.modificarEmpleado()
            elif option == 4:
                workers.despedir_empleado()
            elif option == 5:
                workers.in_out()            
            elif option == 0:
                print("Saliendo!")
                break
            else:
                print("opcion invalida")