#date: 2024-02-06T16:57:00Z
#url: https://api.github.com/gists/207f20d63159d0e700b20d8661c7c77e
#owner: https://api.github.com/users/NlJrGA

# Para este primer reto de la semana, tu objetivo será poder crear un programa en Python el cual permita registrar a un usuario en el sistema.
# Para ello el programa deberá pedir a nuestro usuario final ingrese su siguiente información.

# Solicitar información al usuario
nombre = input("Ingrese su(s) nombre(s): ")
apellidos = input("Ingrese su(s) apellidos: ")
telefono = input("Ingrese su número de teléfono: ")
correo_electronico = input("Ingrese su correo electrónico: ")

# Mensaje de bienvenida
print("\nHola " + nombre + " " + apellidos + ", en breve recibirás un correo a " + correo_electronico +".")