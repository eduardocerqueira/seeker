#date: 2025-05-27T16:50:59Z
#url: https://api.github.com/gists/aae284f2446adf00c0bc240d2d43cf33
#owner: https://api.github.com/users/zanfranceschi

import math

# Introdução ao RSA
# Material prático do conteúdo em https://dev.to/zanfranceschi/uma-introducao-simples-ao-rsa-gek

# dois números primos
p = 11
q = 13

# produto de p e q
n = p * q

# função totiente de Euler
phi = (p - 1) * (q - 1)

# chave pública
e = 17

# chave privada
d = 113

# verifica se e e phi são coprimos
assert math.gcd(e, phi) == 1, "`e` e `phi` não são coprimos"

# verifica se e e d são inversos módulo phi
assert (e * d) % phi == 1, "`e` e `d` não são inversos módulo `phi`"

def criptografar(texto, e, n):
    return [pow(ord(char), e) % n for char in texto]

def descriptografar(texto_criptografado, d, n):
    return "".join([chr(pow(char, d) % n) for char in texto_criptografado])

def texto_criptografado_para_string(texto_criptografado):
    return "".join([chr(c) for c in texto_criptografado])

texto = "teste 1, 2, 3..."
texto_criptografado = criptografar(texto, e, n)
texto_descriptografado = descriptografar(texto_criptografado, d, n)

print("Introdução ao RSA")
print(" texto aberto           > ", texto)
print(" texto criptografado    > ", texto_criptografado_para_string(texto_criptografado))
print(" texto descriptografado > ", texto_descriptografado)

