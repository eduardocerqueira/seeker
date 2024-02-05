#date: 2024-02-05T16:50:33Z
#url: https://api.github.com/gists/7aef213963e1e31ceff2812e61eb4f7c
#owner: https://api.github.com/users/ArmandoRodMen

nombre = input("Ingrese nombre: ")
apellidos = input("Ingrese apellidos: ")
telefono = input("Ingrese teléfono: ")
correo = input("Ingrese correo: ")

telefono = int(telefono) 

#print(nombre, type(nombre), apellidos, type(apellidos), telefono, type(telefono), correo, type(correo))
print("Hola "+nombre+" "+apellidos+", en breve recibirás un correo a "+correo)