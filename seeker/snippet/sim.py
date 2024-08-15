#date: 2024-08-15T17:02:58Z
#url: https://api.github.com/gists/41aaf4253193163fa48f0cdc410f73af
#owner: https://api.github.com/users/FBDev64

# Initialisation
argent = 20
semaine = 0

# Boucle de simulation
while True:
    # Grand-père donne 10 $
    argent += 10

    # Soeur vole 3 $
    if semaine % 2 == 0:
        argent -= 3

    # Incrémente la semaine
    semaine += 1

    # Affiche la quantité d'argent
    print(f"Après {semaine} semaine(s), vous avez {argent} $")

    # Demande à l'utilisateur s'il veut continuer
    continuer = input("Voulez-vous continuer ? (o/n) ")
    if continuer == "n":
        break
