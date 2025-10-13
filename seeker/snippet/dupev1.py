#date: 2025-10-13T16:52:52Z
#url: https://api.github.com/gists/aaa1958d82ca9219c842c71694ad4461
#owner: https://api.github.com/users/ItsCogo-lab

# Gestor d'arxius Duplicats v1.0 -> Començat a les 12:00PM de Dimarts 23/07/2025
# Primera sessió: 12:00PM -> 12:50PM, 23/07/25 (50 Minuts)
# Segona sessió 13:30 -> 14:45, 23/07/25 (75 Minuts)
# FINALITZAT en 125 minuts
# Primera Revisió: 14:45 -> 14:55, 23/07/25 (10 Minuts)
#   PROPOSTES: MILLORES D'OPTIMITZACIÓ
#       Cal un métode de comparació millor, molt lent en aquest cas
#       No es fa servir hash:
#           O el faig servir a la comparació o no el calculo
#       Al calcular hash, agafa tot el codi:
#           Pot complicar-se en arxius grans
#           Cal limitar hash a 64KB/arxiu
#   PROPOSTES: MILLORES D'ÚS
#       Agafar ruta fent servir os (en cas de fer servir en Linux o MacOS)

# Us de IA (mes o menys)
#   ChatGPT: 10 preguntes
#   Gemini: Revisió + 3 Preguntes
#   2 C&P sense entendre, 3 C&P entesos

# Aquest és literalment el primer projecte que faig "sol"

import os
import hashlib
LONGITUD_MAXIMA = 50


def obtener_carpeta():
    ruta = input(r"Introduce el directorio de la carpeta que quieres comprobar los archivos duplicados: ")
    if ruta[-1] != "/":
        ruta = f"{ruta}/"
    return ruta

def lista_de_archivos(ruta): # Muestra una lista de archivos (da igual si son duplicados o no) de toda la carpeta
    return [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))] # IA: C&P, MoM E

def recorrer_lista(ruta, lista_archivos):
    diccionario = []
    for i in lista_archivos:
        ruta_archivo = f"{ruta}{i}"
        a = os.path.getsize(ruta_archivo)
        with open(ruta_archivo, "rb") as f:
            contenido = f.read()
            hash = hashlib.md5(contenido).hexdigest() # IA: C&P, E
        diccionario.append({"archivo": f"{i}", "tamaño": a, "hashlib": hash})
    return diccionario

def print_dict(diccionario):
    print("Archivos en esta carpeta:")
    diccionario_ordenado = sorted(diccionario, key = lambda x: x["tamaño"], reverse = True) # IA: E 
    for i in diccionario_ordenado:
        extension = os.path.splitext(i["archivo"])
        print(f"    {i["archivo"][0:LONGITUD_MAXIMA].rstrip()}..{extension[1]} ocupa {i["tamaño"]} bytes. Su hashlib es {i["hashlib"]}" if len(i["archivo"])>35 
              else f"   {i["archivo"]} ocupa {i["tamaño"]} bytes. Su hashlib es {i["hashlib"]}")

def detectar_duplicados(diccionario): # Molt ineficient, numero de comparacions = (n**2), on n = nombre d'arxius
    lista_dupes = []
    for a in range(len(diccionario)):
        for a2 in range(a+1, len(diccionario)): # IA: C&P, E, 
            if diccionario[a]["tamaño"] == diccionario[a2]["tamaño"] and diccionario[a]["archivo"] != diccionario[a2]["archivo"]:
                print(f"{diccionario[a]["archivo"]} y {diccionario[a2]["archivo"]} son duplicados.")
                lista_dupes.append({"archivo1": diccionario[a]["archivo"], 
                                    "archivo2": diccionario[a2]["archivo"], 
                                    "tamaño": diccionario[a]["tamaño"]})
    return lista_dupes

def convertir_tamaño(lista_duplicados):
    for a in lista_duplicados:
        if a["tamaño"] < 1024:
            # bytes
            pass
        elif a["tamaño"] < 1024**2:
            # kilobytes
            a["tamaño"] = round(a["tamaño"] / 1024, 2)
            a["unidad"] = "KB"
        elif a["tamaño"] < 1024**3:
            # megabytes
            a["tamaño"] = round(a["tamaño"] / 1024**2, 2)
            a["unidad"] = "MB"
        else:
            # gigabytes
            a["tamaño"] = round(a["tamaño"] / 1024**3, 2)
            a["unidad"] = "GB"
    return lista_duplicados
        


def borrar_duplicados(lista_duplicados, ruta, diccionario):
    while len(lista_duplicados) > 0:
        for a in lista_duplicados:
            count = 1
            print(f"Caso {count}:")
            print(f"    Archivo 1: {a["archivo1"]}")
            print(f"    Archivo 2: {a["archivo2"]}")
            print(f"    Tamaño de cada uno: {a["tamaño"]}{a["unidad"]}")
            quiere_borrar = input("Quieres borrar uno de los dos archivos? a1/a2/n ")
            if quiere_borrar == "a1":
                d = next(item for item in diccionario if item["archivo"] == f"{a["archivo1"]}") # IA, C&P
                os.remove(f"{ruta}{d["archivo"]}")
                print(f"Se ha borrado {d["archivo"]}")
            elif quiere_borrar == "a2":
                d = next(item for item in diccionario if item["archivo"] == f"{a["archivo2"]}") # IA, C&P
                os.remove(f"{ruta}{d["archivo"]}")
                print(f"Se ha borrado {d["archivo"]}")
            else:
                print("No se ha borrado ningún archivo.")
            count += 1
            lista_duplicados.remove(a)

def main():
    # ruta = obtener_carpeta()
    ruta = obtener_carpeta()
    print(ruta)
    lista = lista_de_archivos(ruta)
    diccionario_tamaño = recorrer_lista(ruta, lista)
    print_dict(diccionario_tamaño)
    lista_duplicados = detectar_duplicados(diccionario_tamaño)
    lista_duplicados = convertir_tamaño(lista_duplicados)
    borrar_duplicados(lista_duplicados, ruta, diccionario_tamaño)
quiere_repetir = "y"
while quiere_repetir == "y":
    main()
    quiere_repetir = input("No quedan archivos duplicados de esta carpeta. Quieres repetir con otra carpeta? y/n ")

