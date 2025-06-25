#date: 2025-06-25T17:01:16Z
#url: https://api.github.com/gists/4e4551458d021343a278a8a3383e3c9c
#owner: https://api.github.com/users/franlop24

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

class DatabaseManager:
    """
    Una clase para gestionar la conexión a la base de datos y ejecutar consultas SQL en bruto.
    """
    def __init__(self, db_url):
        """
        Inicializa el DatabaseManager.

        Args:
            db_url (str): La URL de conexión a la base de datos (ej. 'sqlite:///mi_base.db',
                          'postgresql://user:pass@host:port/dbname').
        """
        self.db_url = db_url
        self.engine = None
        self._connect()

    def _connect(self):
        """
        Establece la conexión al motor de la base de datos.
        """
        try:
            self.engine = create_engine(self.db_url)
            # Prueba la conexión
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            print(f"Conexión a la base de datos establecida: {self.db_url}")
        except SQLAlchemyError as e:
            print(f"Error al conectar a la base de datos: {e}")
            self.engine = None # Asegurarse de que el motor no esté inicializado si falla la conexión

    def execute_query(self, sql_query, params=None):
        """
        Ejecuta una consulta SQL que no devuelve resultados (INSERT, UPDATE, DELETE).

        Args:
            sql_query (str): La cadena de la consulta SQL.
            params (dict, optional): Un diccionario de parámetros para la consulta. Por defecto es None.

        Returns:
            bool: True si la consulta se ejecutó con éxito, False en caso contrario.
        """
        if not self.engine:
            print("Error: No hay conexión a la base de datos.")
            return False

        try:
            with self.engine.connect() as connection:
                connection.execute(text(sql_query), params or {})
                connection.commit()
            return True
        except SQLAlchemyError as e:
            print(f"Error al ejecutar la consulta: {e}")
            return False

    def fetch_all(self, sql_query, params=None):
        """
        Ejecuta una consulta SQL que devuelve múltiples filas (SELECT).

        Args:
            sql_query (str): La cadena de la consulta SQL.
            params (dict, optional): Un diccionario de parámetros para la consulta. Por defecto es None.

        Returns:
            list: Una lista de diccionarios, donde cada diccionario representa una fila
                  (nombre de columna: valor). Retorna una lista vacía en caso de error o sin resultados.
        """
        if not self.engine:
            print("Error: No hay conexión a la base de datos.")
            return []

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query), params or {})
                # Convierte cada fila a un diccionario para facilitar el acceso
                return [row._asdict() for row in result.mappings()]
        except SQLAlchemyError as e:
            print(f"Error al recuperar datos: {e}")
            return []

    def fetch_one(self, sql_query, params=None):
        """
        Ejecuta una consulta SQL que devuelve una sola fila (SELECT).

        Args:
            sql_query (str): La cadena de la consulta SQL.
            params (dict, optional): Un diccionario de parámetros para la consulta. Por defecto es None.

        Returns:
            dict or None: Un diccionario que representa la fila, o None si no se encontró ninguna fila
                          o si ocurrió un error.
        """
        if not self.engine:
            print("Error: No hay conexión a la base de datos.")
            return None

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query), params or {})
                row = result.mappings().first()
                return row._asdict() if row else None
        except SQLAlchemyError as e:
            print(f"Error al recuperar un solo registro: {e}")
            return None

    def close(self):
        """
        Cierra la conexión al motor de la base de datos.
        (Generalmente no es necesario llamar explícitamente a esto a menos que la aplicación se cierre,
         ya que las conexiones se gestionan con el pool de SQLAlchemy).
        """
        if self.engine:
            self.engine.dispose()
            self.engine = None
            print("Conexión a la base de datos cerrada.")