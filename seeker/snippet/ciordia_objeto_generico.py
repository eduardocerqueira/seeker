#date: 2022-09-26T17:17:29Z
#url: https://api.github.com/gists/454a0d1942e128cbe13efe805d35d871
#owner: https://api.github.com/users/gadiazsaavedra

"""definimos una clase generica"""


class Atleta:
    """definir la clase del atleta"""

    def __init__(self, nombre, apellido, telefono, altura, peso) -> str:
        """definimos atleta"""
        self.nombre = nombre
        self.__apellido = apellido
        self.__telefono = telefono
        self.altura = altura
        self.peso = peso

    def i_m_c(self) -> str:
        """definimos el imc"""
        imc = round((self.peso / self.altura**2), 2)

        if imc < 18.5:
            print(f'imc= {imc} indica: "Peso Inferior"')
        elif imc > 18.5 and imc < 25:
            print(f'imc= {imc} indica: "Peso normal"')
        elif imc > 25 and imc < 29.9:
            print(f'imc= {imc} indica: "Sobrepeso"')
        elif imc > 30 and imc < 34.9:
            print(f'imc= {imc} indica: "Obesidad"')
        else:
            print("Mayor a 35: Obesidad extrema")


_atleta_1 = Atleta("Juan", "Perez", "666666666", 1.80, 80)

print(f"Nombre : {_atleta_1.nombre}")
print(f"Altura : {_atleta_1.altura}")
print(f"Peso : {_atleta_1.peso}")
_atleta_1.i_m_c()
