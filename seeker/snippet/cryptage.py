#date: 2023-05-22T16:36:01Z
#url: https://api.github.com/gists/df2119e128c7d8c6245642643528874e
#owner: https://api.github.com/users/MAHAMAT-AHMAT

tableauSaisie = []
tableauLettre = []
tableauCrypto = []
toto = ()
titi = ()
tab = ()


def saisieTableauLettre(tab):
    """initialisation tableau lettre"""
    for i in range(65, 91):
        z = chr(i)
        tab.append(z)
    return tab


# Creation du tableau de reference
saisieTableauLettre(tableauLettre)


def saisieTableauCrypto(tableauCrypto):
    """initialisation tableau lettre crypto"""
    for i in range(66, 91):
        c = chr(i)
        tableauCrypto.append(c)
    tableauCrypto.insert(25, "A")
    return tableauCrypto


# Creation du tableau crypte
saisieTableauCrypto(tableauCrypto)


def saisieTableauSaisie(tab):
    """saisie du tableau de depart """
    for i in range(taille):
        x = str(input("Saisir la lettre à ajouter dans le tableau : "))
        tab.append(x)
    return tab


def Crypt(tab):
    """changement de lettre Cryptage"""
    for i in range(0, taille):
        for j in range(0, 26):
            if tableauSaisie[i] == tableauLettre[j]:
                l = tableauCrypto[j]
                tab.append(l)
                j = 26
    return tab


def Decrypt(tab):
    """changement de lettre Decryptage """
    for i in range(0, taille):
        for j in range(0, 26):
            if toto[i] == tableauCrypto[j]:
                m = tableauLettre[j]
                tab.append(m)
                j = 26
    return tab


# affichage final

# programme principal

# nombre de lettre du mot à saisir
taille = int(input("Entrer le nombre de lettre à saisir : "))
# Mise en memoire du tableau saisie
tableauSaisie = saisieTableauSaisie(tableauSaisie)
# Mise en memoire du Cryptage
Crypt = Crypt(toto)
# Affichage
print("le message ", tableauSaisie, "crypté est ", Crypt)
print("le message décrypté de ", Crypt, "est ", Decrypt(titi))