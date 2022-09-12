#date: 2022-09-12T16:55:30Z
#url: https://api.github.com/gists/add74dc50777f66b17413bd51ce0ab8b
#owner: https://api.github.com/users/rodrigo-zamora

from __future__ import annotations
from enum import Enum, unique

class Mesa:

    def __init__(self, numeroComensales: int, numeroMesa: int, pedido: Pedido) -> None:
        self._numeroComensales = numeroComensales
        self._numeroMesa = numeroMesa
        self.pedido = pedido

    @property
    def numeroComensales(self) -> int:
        return self._numeroComensales

    @numeroComensales.setter
    def numeroComensales(self, numeroComensales: int) -> None:
        self._numeroComensales = numeroComensales

    @property
    def numeroMesa(self) -> int:
        return self._numeroMesa

    @numeroMesa.setter
    def numeroMesa(self, numeroMesa: int) -> None:
        self._numeroMesa = numeroMesa

    def __str__(self) -> str:
        return f'La mesa {self.numeroMesa} tiene {self.numeroComensales} comensales y su pedido es {self.pedido}'


class Pedido:

    def __init__(self, productos: list) -> None:
        self._productos = productos

    @property
    def productos(self) -> list:
        return self._productos

    @productos.setter
    def productos(self, productos: list) -> None:
        self._productos = productos

    def cuenta(self, numeroPersonas: int) -> int:
        '''Este método sirve para calcular el total de la cuenta.
        El total de la cuenta puede ser dividido entre el número de personas'''

        total = 0

        for producto in self._productos:
            total += (producto.precio - (producto.precio * producto.descuento))

        return total / numeroPersonas

    def agregarProducto(self, producto: Producto) -> None:
        '''Este método sirve para agregar un producto al pedido'''
        self._productos.append(producto)

    def eliminarProducto(self, producto: Producto) -> None:
        '''Este método sirve para eliminar un producto del pedido'''
        self._productos.remove(producto)

    def __str__(self) -> str:
        toReturn = ''
        for producto in self._productos:
            toReturn += f'{producto}, '
        return toReturn


class Producto():

    def __init__(self, nombre: str, descuento: float, precio: float, categoria: str) -> None:
        self._nombre = nombre
        self._descuento = descuento
        self._precio = precio
        self._categoria = categoria

    @property
    def nombre(self) -> str:
        return self._nombre

    @nombre.setter
    def nombre(self, nombre: str) -> None:
        self._nombre = nombre

    @property
    def descuento(self) -> float:
        return self._descuento

    @descuento.setter
    def descuento(self, descuento: float) -> None:
        self._descuento = descuento

    @property
    def precio(self) -> str:
        return self._precio

    @precio.setter
    def precio(self, precio: str) -> None:
        self._precio = precio

    @property
    def categoria(self) -> float:
        return self._categoria

    @categoria.setter
    def categoria(self, categoria: float) -> None:
        self._categoria = categoria
        
    def __str__(self) -> str:
        return f'El producto {self.nombre} de categoria {self.categoria} tiene un descuento de {self.descuento} y un precio de {self.precio}'

class Empleado:

    def __init__(self, nombre: str) -> None:
        self._nombre = nombre

    @property
    def nombre(self) -> str:
        return self._nombre

    @nombre.setter
    def nombre(self, nombre: str) -> None:
        self._nombre = nombre

    def trabajar(self) -> str:
        '''Este metodo sirve para que el empleado trabaje'''
        return f"{self.nombre} se pone a trabajar"

class Mesero(Empleado):

    def __init__(self, nombre: str, mesas: list) -> None:
        super().__init__(nombre)
        self._mesas = mesas

    @property
    def mesas(self) -> list:
        return self._mesas

    @mesas.setter
    def mesas(self, mesas: list) -> None:
        self._mesas = mesas

    def trabajar(self) -> str:
        '''Este metodo sirve para que el mesero trabaje'''
        return f"{self.nombre} se pone a trabajar como mesero"

    def agregarMesa(self, mesa: Mesa) -> None:
        '''Este metodo sirve para agregar una mesa al mesero'''
        self._mesas.append(mesa)

    def quitarMesa(self, mesa: Mesa) -> None:
        '''Este metodo sirve para quitar una mesa al mesero'''
        self._mesas.remove(mesa)

    def __str__(self) -> str:
        return f'El mesero {self.nombre} tiene {self.mesas} mesas'


class Menu:
    
    @unique
    class Categoria(Enum):
        DESAYUNOS = 1
        POSTRES   = 2
        ENSALADAS = 3
        BEBIDAS   = 4
        ALIMENTOS = 5
    
    def __init__(self,  productos:list) -> None:
        ''' Retorna una nueva instancia de menú '''
        
        self.productos = productos
        
    def __str__(self) -> str:
        toReturn = 'El menú tiene los siguientes productos:\n'
        for producto in self.productos:
            toReturn += f'- {producto}\n'
        return toReturn

if __name__ == '__main__':

    jericalla = Producto("Jericalla", 0, 20, Menu.Categoria.POSTRES)
    refresco = Producto("Refresco", 0, 30, Menu.Categoria.BEBIDAS)
    hamburguesa = Producto("Hamburguesa", 0, 70, Menu.Categoria.ALIMENTOS)
    
    menu = Menu([jericalla, refresco, hamburguesa])
    
    print(menu, '\n')

    mesa1 = Mesa(4, 1, Pedido([]))
    mesa1.pedido.agregarProducto(jericalla)
    mesa1.pedido.agregarProducto(refresco)
    mesa1.pedido.agregarProducto(hamburguesa)
    
    print(mesa1, '\n')

    print('Cada comensal de la mesa debe pagar:', mesa1.pedido.cuenta(4), '\n')