#date: 2024-02-08T17:10:01Z
#url: https://api.github.com/gists/0ca61bd6ce22cd6c11856f4853a8c29d
#owner: https://api.github.com/users/llontopdev

MIN_LEN_TEXTO=5
MAX_LEN_TEXTO=50
NORMAL_LEN_TEL=10

registro = []

print('Registro de Usuarios')

cantidad = input('Ingrese cantidad de usuarios a registrar: ')

while not cantidad.isdigit():
  print('la cantidad debe ser un numero mayor a 0')
  cantidad = input('Ingrese cantidad de usuarios a registrar: ')

veces = int(cantidad)
for i in range(1, veces+1):
  print('Registro Usuario', i)

  nombre = input('Ingrese su nombre: ')
  while len(nombre) < MIN_LEN_TEXTO or len(nombre) > MAX_LEN_TEXTO:
    print('El campo "nombre" debe tener 5 caracteres como mínimo y 50 como máximo')
    nombre = input('Ingrese su nombre: ')

  apellido = input('Ingrese su apellido: ')
  while len(apellido) < MIN_LEN_TEXTO or len(apellido) > MAX_LEN_TEXTO:
    print('El campo "apellido" debe tener 5 caracteres como mínimo y 50 como máximo')
    apellido = input('Ingrese su apellido: ')

  telefono = input('Ingrese su telefono: ')
  while len(telefono) != NORMAL_LEN_TEL or not telefono.isdigit():
    print('El campo "telefono" debe ser numerico y debe tener 10 digitos')
    telefono = input('Ingrese su telefono: ')

  email = input('Ingrese su email: ')
  while len(email) < MIN_LEN_TEXTO or len(email) > MAX_LEN_TEXTO:
    print('El campo "email" debe tener 5 caracteres como mínimo y 50 como máximo')
    email = input('Ingrese su email: ')
  id = len(registro)+1
  registro.append(id)
  print('usuario',id,'registrado correctamente')
  
print(registro)