#date: 2024-02-08T17:08:25Z
#url: https://api.github.com/gists/f605c883193f7b3dd0fe43cadd0a5de0
#owner: https://api.github.com/users/w2k31984

def validar_longitud(cadena, minimo, maximo):
  """
  Valida si la longitud de una cadena está dentro de un rango.

  Args:
      cadena: La cadena a validar.
      minimo: La longitud mínima permitida.
      maximo: La longitud máxima permitida.

  Returns:
      True si la longitud de la cadena está dentro del rango, False en caso contrario.
  """
  return minimo <= len(cadena) <= maximo

def validar_telefono(numero):
  """
  Valida si un número de teléfono tiene 10 dígitos.

  Args:
      numero: El número de teléfono a validar.

  Returns:
      True si el número de teléfono tiene 10 dígitos, False en caso contrario.
  """
  return len(numero) == 10 and numero.isdigit()

def registrar_usuario(nombres, apellidos, nTelefono, correoElectronico):
  """
  Registra un nuevo usuario en el sistema.

  Args:
      nombre: El nombre del usuario.
      apellidos: Los apellidos del usuario.
      telefono: El número de teléfono del usuario.
      email: El correo electrónico del usuario.
  """
  # Validaciones
  if not validar_longitud(nombres, 5, 50):
    print("El nombre debe tener entre 5 y 50 caracteres.")
    return
  if not validar_longitud(apellidos, 5, 50):
    print("Los apellidos deben tener entre 5 y 50 caracteres.")
    return
  if not validar_longitud(correoElectronico, 5, 50):
    print("El correo electrónico debe tener entre 5 y 50 caracteres.")
    return
  if not validar_telefono(nTelefono):
    print("El número de teléfono debe tener 10 dígitos.")
    return

  # Registro del usuario
  print("Usuario registrado correctamente.")
  # Aquí se debería implementar el código para almacenar la información del usuario en una base de datos.

# Registro de usuarios
numero_usuarios = int(input("¿Cuántos usuarios desea registrar?: "))

for i in range(numero_usuarios):
#Declaramos las variables que capturan los datos
 nombres   = input('Ingresa tu nombre: ')
 apellidos = input('Ingresa tu apellido: ')
 nTelefono = int(input('Ingresa tu numero de telefono: '))
 correoElectronico = input('Ingresa tu correo electronico: ')

registrar_usuario(nombres, apellidos, nTelefono, correoElectronico)