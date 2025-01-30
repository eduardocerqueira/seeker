#date: 2025-01-30T16:49:10Z
#url: https://api.github.com/gists/aeac9b0fd868e3a836d5827576966c71
#owner: https://api.github.com/users/Danny-Verdugo

# Codigo Morse

# Letras del alfabeto
alfabeto = "abcdefghijklmnopqrstuvwxyz"

# Dicionario con las claves del codigo morse
morse_code = { 'A':'.-', 'B':'-...',
                    'C':'-.-.', 'D':'-..', 'E':'.',
                    'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-',
                    'L':'.-..', 'M':'--', 'N':'-.',
                    'O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-',
                    'U':'..-', 'V':'...-', 'W':'.--',
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--',
                    '4':'....-', '5':'.....', '6':'-....',
                    '7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ', ':'--..--', '.':'.-.-.-',
                    '?':'..--..', '/':'-..-.', '-':'-....-',
                    '(':'-.--.', ')':'-.--.-'}

def codificar_mensaje(mensaje):
    # string vacio para agregar las claves codificadas
    mensaje_codificado = ""

    # iteramos el mensaje a codificar
    for letra in mensaje:
      # valida si existe la letra en el alfabeto
        if letra.upper() in alfabeto.upper():
          # agrega el valor del codigo a la variable
            mensaje_codificado += morse_code[letra.upper()] + "/"
          # agrega la letra del mensaje
            mensaje_codificado += letra + "/"
    # retorna el mensaje ya codificado
    return mensaje_codificado

# Escribe el mensaje a codificar
messeger = input("Escribe un mensaje: ")

# guarda el resusltado de la funcion en la variable resultado
resultado = codificar_mensaje(messeger)
# imprime en consola el resultado
print(resultado)