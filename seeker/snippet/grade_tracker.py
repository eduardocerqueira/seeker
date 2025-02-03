#date: 2025-02-03T17:07:15Z
#url: https://api.github.com/gists/bdcb98de6d93b94ed64180e4ebd8e4ac
#owner: https://api.github.com/users/Danny-Verdugo

#  crear la clase estudiante
class Estudiante:

  # crear el metodo init con el atributo nombre
  def __init__(self, nombre):
    self.nombre = nombre
    self.grades = {"matematicas":0, "ciencias": 0, "historia":0}

  # creamos el metodo actualiza calificacion
  def update_grade(self, subject, grade):
    self.grades[subject] = grade

  # crear el metodo string para representar los datos del estudiante
  def __str__(self):
    return f"Nombre: {self.nombre} Notas: matematicas={self.grades['matematicas']}, ciencias={self.grades['ciencias']}, historia={self.grades['historia']}"

# crear la clase el libro de calificaciones
class Gradebook:

  # crear el metodo init 
  def __init__(self): 
    self.estudiantes = {}

  # crear el metodo para agregar estudiante al libro de calificaciones
  def add_estudiante(self, nombre):
    estudiante =Estudiante(nombre)
    self.estudiantes[nombre] = estudiante

  # crear el metodo para eliminar estudiante al libro de calificaciones
  def remove_estudiante(self, nombre):
    if nombre in self.estudiantes:
      del self.estudiantes[nombre]
      return True
    else:
      return False

  
  # crear el metodo para actualizar estudiante al libro de calificaciones
  def update_grade(self, nombre, subject, grade):
    if nombre in self.estudiantes:
      estudiante = self.estudiantes[nombre]
      estudiante.update_grade(subject, grade)
      return True
    else:
      return False

  # crear el metodo string para representar el libro de calificaciones
  def __str__(self):
    resultado = "Alumnos actuales:\n"
    for name,estudiante in self.estudiantes.items():
      resultado += str(estudiante) + "\n"
    return resultado

# creamos la instancia del libro de calificaciones
grade_book = Gradebook()

# iniciar el bucle principal del programa, donde el usuario puede realizar
# operaciones en el libro de calificaciones. Ejecutar este bucle hasta que el usuario elija
# salir del programa.
salir = False

while not salir:
# Imprimir las opciones que el usuario puede elegir:
# 1. Agregar un estudiante.
# 2. Actualizar la calificación del estudiante.
# 3. Eliminar un estudiante.
# 4. Mostrar todos los estudiantes.
# 5. Salir.
  print("\n")
  print("Elige uan opciones:")
  print("1. Agregar un estudiante")
  print("2. Actualizar la calificación del estudiante")
  print("3. Eliminar un estudiante")
  print("4. Mostrar todos los estudiantes")
  print("5. Salir")
    
# Tome la elección del usuario ('1', '2', '3', '4' o '5'. Cadena).
  elegir = input("Qué es lo que quieres hacer: ")
  print("\n")
  
  # Si la opción es '1':
  if elegir == "1":

    # Tome el nombre de un estudiante del usuario, agréguelo al
    # libro de calificaciones e imprima un mensaje de confirmación.
    nombre = input("Ingrese el nombre del estudiante: ")
    grade_book.add_estudiante(nombre)
    print(f"{nombre} ha sido agregado al libro de calificaciones")
  
  # Si la opción es '2':
  elif elegir == "2":

    # Tomar el nombre, la materia y la calificación de un estudiante del usuario y actualizar el libro de calificaciones.
    # Tomar el nombre de un estudiante del usuario y eliminarlo del libro de calificaciones.
    # Mostrar un mensaje de éxito/fracaso según el resultado de la operación.
    nombre = input("Ingrese el nombre del estudiante: ")
    subject = input("Ingrese la materia: (matematicas, ciencias o historia)")
    grade = int(input("Ingrese la calificación: "))
    if grade_book.update_grade(nombre, subject, grade):
      print(f"La calificación de {nombre} en la materia {subject} ahora tiene la calificación de {grade}")
    else:
      print(f"{nombre} no se encuentra en el libro de calificaciones")

  # Si la opción es '3':
  elif elegir == "3":

      
      # Tome el nombre de un estudiante del usuario y elimínelo del libro de calificaciones.
      # Muestre un mensaje de éxito/fracaso según el resultado de la operación.
      nombre = input("Ingrese el nombre del estudiante: ")
      if grade_book.remove_estudiante(nombre):
        print(f"{nombre} ha sido eliminado del libro de calificaciones")
      else:
        print(f"{nombre} no se encuentra en el libro de calificaciones")

  # Si la opción es '3':
  elif elegir == "4":

      # imprime el libro de calificaciones
      print(grade_book)

  # Si la opción es '5':
  elif elegir == "5":

      # Salir del programa
      salir = True


  # Si la elección no es válida, muestra un mensaje de error y vuelve a preguntar al usuario.
  else:
    print("Opción no válida. Por favor, elige una opción válida.")