#date: 2026-01-28T17:16:40Z
#url: https://api.github.com/gists/acdc8198386974ea67874a407f38d02c
#owner: https://api.github.com/users/Edu2610

"""
EVIDENCIA TEMAS 8 y 9: DICCIONARIOS Y CONJUNTOS
Autor: Eduardo
Descripción: Análisis de listas de correos usando Teoría de Conjuntos y Hash Maps.
"""

# ==========================================
# PARTE 1: DATOS CRUDOS (Listas con duplicados)
# ==========================================
curso_python = ["juan@u.cl", "ana@u.cl", "juan@u.cl", "pedro@u.cl"]
curso_sql    = ["ana@u.cl", "maria@u.cl", "pedro@u.cl"]

print(f"Inscritos Python (Bruto): {len(curso_python)}")
print(f"Inscritos SQL (Bruto): {len(curso_sql)}")

# ==========================================
# PARTE 2: LÓGICA DE CONJUNTOS (Sets)
# ==========================================

# 1. Limpieza (Eliminar duplicados automáticamente)
set_python = set(curso_python)
set_sql    = set(curso_sql)

# 2. Operaciones de Conjuntos
# Intersección (&): Alumnos en AMBOS cursos (nerds)
doble_matricula = set_python & set_sql

# Unión (|): Total de alumnos únicos en la academia
total_alumnos = set_python | set_sql

# Diferencia (-): Alumnos SOLO en Python (que no están en SQL)
solo_python = set_python - set_sql

# ==========================================
# PARTE 3: REPORTE (Diccionario Avanzado)
# ==========================================

reporte = {
    "metricas": {
        "total_unicos": len(total_alumnos),  # Corregido: len() en vez de .length()
        "cantidad_dobles": len(doble_matricula)
    },
    "listas_limpias": {
        "doble_matricula": doble_matricula,
        "solo_python": solo_python
    }
}

# Uso de .get() para acceso seguro (Tema 8)
# Si pedimos una clave que no existe, no explota, devuelve el valor por defecto.
print("\n--- REPORTE FINAL ---")
print(f"Total Alumnos: {reporte['metricas']['total_unicos']}")
print(f"Alumnos Dobles: {reporte['metricas']['cantidad_dobles']}")
print(f"Correos Dobles: {reporte['listas_limpias']['doble_matricula']}")

# Prueba de seguridad
print(f"Campo inexistente: {reporte.get('errores', 'No hay errores reportados')}")


# ==========================================
# PARTE 4: CHEAT SHEET (SETS & DICT AVANZADO)
# ==========================================

"""
1. CONJUNTOS (SETS) -> { val1, val2 }
   - Colección desordenada de elementos ÚNICOS.
   - set(lista) -> Truco rápido para borrar duplicados.
   
   OPERACIONES LÓGICAS:
   A & B  -> Intersección (Elementos en A Y B).
   A | B  -> Unión (Elementos en A O B).
   A - B  -> Diferencia (Están en A pero NO en B).
   A ^ B  -> Diferencia Simétrica (En A o B, pero NO en ambos).

2. DICCIONARIOS PRO
   - Acceso Anidado: dic["clave1"]["subclave2"]
   - Acceso Seguro:  dic.get("clave", "Valor Por Defecto")
     ¡Úsalo siempre que leas datos externos (JSON/APIs)!
     
   - .keys()   -> Lista de claves.
   - .values() -> Lista de valores.
   - .items()  -> Lista de tuplas (clave, valor).
"""