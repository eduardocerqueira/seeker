#date: 2022-07-29T17:08:54Z
#url: https://api.github.com/gists/f3fcaedc4c8cdb681041808261f715d6
#owner: https://api.github.com/users/StevensRemache

class Material(object):
    """
    Clase padre encargada de heredar a las clases hijas - subclases(Libro-Revista)
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

    def __init__(self, codigo: str = None, autor: str = None, titulo: str = None, anio: int = None,
                 editorial: str = None, disponible: bool = True, cantidad_disponible: int = None) -> object:
        self._codigo = codigo
        self._autor = autor
        self._titulo = titulo
        self._anio = anio
        self._editorial = editorial
        self._disponible = disponible
        self._cantidad_disponible = cantidad_disponible

    def __str__(self):
        return f'Material[Código: {self._codigo}, Autor: {self._autor}, Título: {self._titulo}, Año: {self._anio},' \
               f' Editorial: {self._editorial}, Disponible: {self._disponible},' \
               f' Cantidad_Disponible: {self._cantidad_disponible}]'

    def imprimir_material(self):
        return f'Material[Código: {self._codigo}, Autor: {self._autor}, Título: {self._titulo}, Año: {self._anio},' \
               f' Editorial: {self._editorial}, Disponible: {self._disponible},' \
               f' Cantidad_Disponible: {self._cantidad_disponible}]'
    @property
    def codigo(self):
        return self._codigo

    @codigo.setter
    def codigo(self, codigo):
        self._codigo = codigo

    @property
    def autor(self):
        return self._autor

    @autor.setter
    def autor(self, autor):
        self._autor = autor

    @property
    def titulo(self):
        return self._titulo

    @titulo.setter
    def titulo(self, titulo):
        self._titulo = titulo

    @property
    def anio(self):
        return self._anio

    @anio.setter
    def anio(self, anio):
        self._anio = anio

    @property
    def editorial(self):
        return self._editorial

    @editorial.setter
    def editorial(self, editorial):
        self._editorial = editorial

    @property
    def disponible(self):
        return self._disponible

    @disponible.setter
    def disponible(self, disponible=False):
        self._disponible = disponible

    @property
    def cantidad_disponible(self):
        return self._cantidad_disponible

    @cantidad_disponible.setter
    def cantidad_disponible(self, cantidad_disponible):
        self._cantidad_disponible = cantidad_disponible

    @staticmethod
    def actualizar_disponibilibad(actualizar_disponibilibad=bool):
        return actualizar_disponibilibad

    pass


if __name__ == '__main__':
    MarCab = Material(codigo='ISBN0-385-50420-9', autor='Dan Brown', titulo='El Código Da Vinci', anio=2003,
                      editorial='Doubleday Transworld Publishers Bantam Books', disponible=True, cantidad_disponible=15)
    print(MarCab)
    print(MarCab.actualizar_disponibilibad(True))
