#date: 2023-01-10T16:59:31Z
#url: https://api.github.com/gists/b39f808a68471c8c9fd529daf0c2bdea
#owner: https://api.github.com/users/SalajanPaul

def parcurgere_lista():
    l1 = list()
    l1 = [1, 2, 3, 4, 5, 7, "d", "c"]
    # for element in l1:
    #     print(l1)
    for i in range(len(l1)):
        print(l1[i])

def sumalitere():
    #dictionar cu acolada
    dict1 = {"a":10, "b":20, "c":300}
    suma = 0
    cuv = "abc"
    for i in range(len(cuv)):
        suma += dict1[cuv[i]]
    print("Suma este:", suma)
    # print(4+2)

def crypt_hash(propozitie):
    dict3crypt = {"a":"b", "b":"e", "c":"f", "d":"g", "e":"h", "f":"i", "g":"j", "h":"k",
                  "i":"l", "j":"m", "k":"n", "l":"o", "m":"p", "n":"q", "o":"r", "p":"s",
                  "q":"t", "r":"u", "s":"v", "t":"w", "u":"x", "v":"y", "w":"z", "x":"a",
                  "y":"b", "z":"c"}
    rezultat = ""
    for element in propozitie:
        if element in (".", "-", "?", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"):
            rezultat += element
        else:
            rezultat += dict3crypt[element]
    return rezultat

def decrypt_hash(propozitie):
    dict3decrypt = {"a":"b", "b":"e", "c":"f", "d":"g", "e":"h", "f":"i", "g":"j", "h":"k",
                  "i":"l", "j":"m", "k":"n", "l":"o", "m":"p", "n":"q", "o":"r", "p":"s",
                  "q":"t", "r":"u", "s":"v", "t":"w", "u":"x", "v":"y", "w":"z", "x":"a",
                  "y":"b", "z":"c"}
    rezultat = ""
    for element in propozitie:
        if element in (".", "-", "?", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"):
            rezultat += element
        else:
            rezultat += dict3decrypt[element]
    return rezultat



if __name__ == "__main__":
    # parcurgere_lista()
    # sumalitere()
    propozitie = input("Introduceti o propozitie: ")
    c1 = crypt_hash(propozitie)
    print(c1)
    c2 = decrypt_hash(c1)
    print(c2)
