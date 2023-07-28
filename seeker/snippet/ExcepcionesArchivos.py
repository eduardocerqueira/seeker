#date: 2023-07-28T16:40:23Z
#url: https://api.github.com/gists/bb512e73a233ba5c683ff6c9217b43b8
#owner: https://api.github.com/users/Kevin-Jimenez-D

try:
    with open("archivo.txt", "r") as file:
        #Manipulación del archivo
except FileNotFoundError:
    print("El archivo no existe")
except Exception:
    print("Error al manejar el archivo")