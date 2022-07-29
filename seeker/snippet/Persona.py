#date: 2022-07-29T17:15:20Z
#url: https://api.github.com/gists/41bca8cab914ca59b6afb899865e12bf
#owner: https://api.github.com/users/StevensRemache

class Persona(object):
    """
    Clase padre encargada de heredar comportamientos a las clases hijas - subclases(Estudiante-Docente)
        Universidad de Guayaquil
        Curso: GIG-S-MA-3-1
        Prof: Ing. Guillermo Valarezo Guzmán
                GRUPO N°4
               INTEGRANTES:
    	Cabeza Angulo Maria del Carmen.
    	Ramón Sesme Angela Leonela.
    	Remache Ochoa Steven Nelson.
    	Sigcho Cueva Geordy Andrés.
    	Tubay Zambrano Nicole Michelle.
    """

    def __init__(self, cedula: str = None, nombre: str = None, apellido: str = None, email: str = None,
                 telefono: str = None, direccion: str = None, numero_libros: int = None,
                 activo: bool = False, carrera: str = None):
        self._cedula = cedula
        self._nombre = nombre
        self._apellido = apellido
        self._email = email
        self._telefono = telefono
        self._direccion = direccion
        self._numero_libros = numero_libros
        self._activo = activo
        self._carrera = carrera

    def __str__(self) -> str:
        return f'Persona [Cedula_Id: {self._cedula}, Nombre: {self._nombre}, Apellido= {self._apellido}, ' \
               f'E-mail: {self._email}, : Telefono: {self._telefono}, Dirección={self._direccion},' \
               f' Cantidad de Libros: {self._numero_libros}, Libro Activo: {self._activo}, Carrera: {self._carrera}]'

    def persona(self):
        return (f'El ciudadano (Persona): {self.apellido} {self.nombre}, con CI: {self.cedula} y correo eléctronico '
                f'{self.email} , activo en el sistema: {self.activo}, \n'
                f'perteneciente a la carrera de {self.carrera} posee {self.numero_libros} libros en su potestad.')

    @property
    def cedula(self):
        return self._cedula

    # Solo Lectura (Read Only) Ya que la cédula no se puede modificar
    # @cedula.setter
    # def cedula(self, cedula):
    #   self._cedula = cedula

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, nombre):
        self._nombre = nombre

    @property
    def apellido(self):
        return self._apellido

    @apellido.setter
    def apellido(self, apellido):
        self._apellido = apellido

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, email):
        self._email = email

    @property
    def telefono(self):
        return self._telefono

    @telefono.setter
    def telefono(self, telefono):
        self._telefono = telefono

    @property
    def direccion(self):
        return self._direccion

    @direccion.setter
    def direccion(self, direccion):
        self._direccion = direccion

    @property
    def numero_libros(self):
        return self._numero_libros

    @numero_libros.setter
    def numero_libros(self, numero_libros):
        self._numero_libros = numero_libros

    @property
    def activo(self):
        return self._activo

    @activo.setter
    def activo(self, activo=False):
        self._activo = activo

    @property
    def carrera(self):
        return self._carrera

    @carrera.setter
    def carrera(self, carrera):
        self._carrera = carrera


if __name__ == '__main__':
    StevOch0 = Persona(cedula='0978998756', nombre='Rosario', apellido='Alvendría', email='rosarito96@gmail.com',
                       direccion='Guayaquil, 098-K6-M10', numero_libros=2, activo=True,
                       carrera='Auxiliar en Enfermería')
    print("\n", StevOch0, "\n")
    print(StevOch0.persona())
