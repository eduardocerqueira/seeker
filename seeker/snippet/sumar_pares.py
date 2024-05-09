#date: 2024-05-09T16:56:56Z
#url: https://api.github.com/gists/0a81a22bb04e7238737df0b63cfa3a75
#owner: https://api.github.com/users/MayeDeveloper

#Sumar pares desde 1 hasta el número ingresado por el usuario
def sumar_pares():
    suma = 0
    numero = int(input("Digite un número: "))
    for i in range(1,numero+1):
        if i % 2 == 0:
            suma = suma + i
    print("La suma de los números pares desde 1 hasta el número ingresado,",numero, "es: ", suma)  
sumar_pares()