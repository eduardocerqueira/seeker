#date: 2023-08-10T16:54:05Z
#url: https://api.github.com/gists/eb88866fd299cde7715e1020776db312
#owner: https://api.github.com/users/JxV0

# Definir las listas l1 y l2
l1 = [("uno", 1), ("dos", 2), ("tres", 3), ("cuatro", 4), ("cinco", 5)] 
l2 = [("uno", 1), ("dos", 2), ("seis", 6)]

# a.Crear conjuntos a partir de las listas
s1 = set(l1)
s2 = set(l2)
# b.Encontrar elementos comunes en s1 y s2
s3 = s1 & s2
# c.Encontrar elementos únicos en s1 y s2
s4 = (s1 - s2) | (s2 - s1)
# d. Crear una nueva lista l3 con elementos de s3 y s4 ordenados por el número entero
l3 = sorted(list(s3) + list(s4), key=lambda x: x[1])
# Imprimir los resultados
print("Conjunto s1:", s1)
print("Conjunto s2:", s2)
print("Conjunto s3 (elementos comunes):", s3)
print("Conjunto s4 (elementos únicos):", s4)
print("Lista l3 (elementos ordenados por número entero):", l3)