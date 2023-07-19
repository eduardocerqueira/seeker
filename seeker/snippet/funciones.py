#date: 2023-07-19T16:58:17Z
#url: https://api.github.com/gists/5ec9d73c385d0cc095421848934f6dbf
#owner: https://api.github.com/users/JuanCarlosMarino

def saludo():#Funcion que no recibe parametros, no retorna nada (si no está la palabra return o si la palabra return está sola no devuelve nada)
    print("Hola a todos!!")
    print(":)")

def saludo2():#Funcion que no recibe parametros, retorna una cadena
    return "Hola a todos!!"

def entrada_permitida(edad):#Función que recibe parametros y retorna un valor
    flag = True
    if edad >= 0 and edad < 18:
        flag = False
    return flag

#Uso de funciones como parámetros
def suma(num1 = 0, num2 = 0):
    return num1+num2

def resta(num1 = 0, num2 = 0):
    return num1-num2

def operacion(operacion_a_ejecutar, num1 = 0, num2 = 0):
    return operacion_a_ejecutar(num1, num2)