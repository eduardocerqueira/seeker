#date: 2026-02-26T17:46:14Z
#url: https://api.github.com/gists/755726de1a781c847ccf2979945bfe1f
#owner: https://api.github.com/users/bormolina

from datetime import datetime

class Producto:
    def __init__(self, id: int, nombre: str, categorias: list[str],
                 precio: float, fecha_entrada: datetime, fecha_caducidad: datetime):
        self.id = id
        self.nombre = nombre
        self.categorias = categorias
        self.precio = precio
        self.fecha_entrada = fecha_entrada
        self.fecha_caducidad = fecha_caducidad

    def __eq__(self, other):
        if not isinstance(other, Producto):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        if not isinstance(other, Producto):
            return NotImplemented
        return self.precio < other.precio

    def __gt__(self, other):
        if not isinstance(other, Producto):
            return NotImplemented
        return self.precio > other.precio

    def __le__(self, other):
        if not isinstance(other, Producto):
            return NotImplemented
        return self.precio <= other.precio

    def __ge__(self, other):
        if not isinstance(other, Producto):
            return NotImplemented
        return self.precio >= other.precio

    def __str__(self):
        return f"{self.nombre} ({self.precio:.2f}€) [id={self.id}]"