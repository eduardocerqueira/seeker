#date: 2025-10-13T16:54:02Z
#url: https://api.github.com/gists/1b17433e3028a0cbee8f832734a8e89d
#owner: https://api.github.com/users/ItsCogo-lab

# Puntuador de contrasenyes

import re
from palabras import palabras_
puntuacion = 0
# Puntuación de contraseñas:
#   Longitud >= 8	+2 fet
#   Longitud >= 12	+2 fet
#   Tiene mayúsculas	+1 fet
#   Tiene minúsculas	+1 fet
#   Tiene dígitos	+1 fet
#   Tiene símbolos especiales	+1 fet
#   No contiene palabras comunes	+2 fet

abecedario_mayus = r"[A-Z]"
abecedario_minus = r"[a-z]"
numeros = r"\d"
caracteres = r"[^\w\s]"

contraseña = str(input("Introduce una contraseña: "))
print(contraseña)
palabras = palabras_()
def tiene_palabra_comun(contraseña, palabras):
    contra = contraseña.lower()
    for p in palabras:
        if len(p) > 4 and p in contra:
            return True, p
    return False, None

def long_contraseña(contraseña):
    longitud = len(contraseña)
    if 12 > longitud >= 8:
        seguridad = "mid"
    elif longitud >= 12:
        seguridad = "top"
    else:
        seguridad = "low"
    return seguridad

seguridad = long_contraseña(contraseña)
if seguridad == "mid":
    print("⚠️ Tu contraseña tiene una longitud mediana. ")
    puntuacion += 2
elif seguridad == "low":
    print("❌ Tu contraseña es demasiada corta. ")
elif seguridad == "top":
    print("✅ Tu contraseña es suficientemente larga!")
    puntuacion += 4

def tiene_numeros_especial(contraseña):
    especial = re.findall(caracteres, contraseña)
    digitos = re.findall(numeros, contraseña)
    if especial and digitos:
        return True, True
    elif especial and not digitos:
        return True, False
    elif not especial and digitos:
        return False, True
    else:
        return False, False

tiene_especiales, tiene_numeros = tiene_numeros_especial(contraseña)
if tiene_especiales and not tiene_numeros:
    print("⚠️ Tu contraseña no tiene dígitos, pero tiene caracteres especiales")
    puntuacion += 1
elif tiene_numeros and not tiene_especiales:
    print("⚠️ Tu contraseña no tiene caracteres especiales, pero tiene dígitos")
    puntuacion += 1
elif tiene_numeros and tiene_especiales:
    print("✅ Tu contraseña tiene dígitos y caracteres especiales")
    puntuacion += 2
elif not tiene_numeros and not tiene_especiales:
    print("❌ Tu contraseña no tiene ni dígitos ni caracteres especiales.")



def tiene_mayusculas_minusculas(contraseña):
    mayus = re.findall(abecedario_mayus, contraseña)
    minus = re.findall(abecedario_minus, contraseña)
    if mayus and minus:
        return True, True
    elif mayus and not minus:
        return True, False
    elif not mayus and minus:
        return False, True
    else:
        return False, False

tiene_mayus, tiene_minus = tiene_mayusculas_minusculas(contraseña)
if tiene_mayus and not tiene_minus:
    print("⚠️ Tu contraseña no tiene minúsculas, pero tiene mayúsculas")
    puntuacion += 1
elif tiene_minus and not tiene_mayus:
    print("⚠️ Tu contraseña no tiene mayúsculas, pero tiene minúsculas")
    puntuacion += 1
elif tiene_minus and tiene_mayus:
    print("✅ Tu contraseña tiene mayúsculas y minúsculas")
    puntuacion += 2
elif not tiene_minus and not tiene_mayus:
    print("❌ Tu contraseña no tiene ni minúsculas ni mayúsculas.")


tiene_palabras_comunes, palabra_comun = tiene_palabra_comun(contraseña, palabras)
if tiene_palabras_comunes == True:
    print(f"⚠️ Tu contraseña tiene al menos una palabra común: {palabra_comun}")
else:
    print("✅ Enhorabuena! Tu contraseña no tiene palabras comunes.")
    puntuacion += 2


print(f"La nota de tu contraseña es {puntuacion}/10!")