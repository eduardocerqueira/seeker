#date: 2024-08-21T17:11:13Z
#url: https://api.github.com/gists/fcf90bfd384e22d87953be553c8ddfb7
#owner: https://api.github.com/users/gitbra

# Jeu facile pour apprendre le Python !
# © 2024, gist.github.com/gitbra

# Constantes du jeu
c_largeur = 7
c_hauteur = 6


# Fonctions réutilisables
def safe_int(column: str) -> int:
    try:
        return int(column)
    except ValueError:              # Tout ce qui n'est pas un chiffre va provoquer une erreur (dit aussi "exception")
        return 0                    # On renvoie alors 0 qui n'est pas entre 1 et 7, donc distinct de la saisie de l'utilisateur


def haswon(grille) -> bool:         # Cette fonction dit si 4 pions sont alignés horizontalement, verticalement ou diagonalement dans la grille passée en paramètre
    # Vérification horizontale
    for y in range(c_hauteur):
        for x in range(c_largeur - 3):
            if grille[y][x] == grille[y][x + 1] == grille[y][x + 2] == grille[y][x + 3] != ' ':
                return True
    # Vérification verticale
    for y in range(c_hauteur - 3):
        for x in range(c_largeur):
            if grille[y][x] == grille[y + 1][x] == grille[y + 2][x] == grille[y + 3][x] != ' ':
                return True
    # Vérification diagonale (haut-gauche vers bas-droit)
    for y in range(c_hauteur - 3):
        for x in range(c_largeur - 3):
            if grille[y][x] == grille[y + 1][x + 1] == grille[y + 2][x + 2] == grille[y + 3][x + 3] != ' ':
                return True
    # Vérification diagonale (bas-gauche vers haut-droit)
    for y in range(c_hauteur - 3):
        for x in range(c_largeur - 3):
            if grille[y + 3][x] == grille[y + 2][x + 1] == grille[y + 1][x + 2] == grille[y][x + 3] != ' ':
                return True
    # Par défaut, on ne gagne pas
    return False


def isfull(grille) -> bool:
    return ' ' not in grille[0]                     # La grille n'est pas pleine si une des colonnes contient une place disponible


def get_symbol(joueur: bool) -> str:                # Cette fonction attribue un symbole à chaque joueur
    return 'O' if joueur else 'X'


# Initialisation de la partie
assert (c_largeur >= 4) and (c_hauteur >= 4)        # Imposé par la mécanique de jeu pour éviter les erreurs de mémoire
grille = []
for _ in range(c_hauteur):
    grille.append([' '] * c_largeur)                # Cette syntaxe évite de dupliquer des références d'objets, chaque ligne est réinstanciée
joueur_un = True

# La partie commence ici !
while True:

    # Demander au joueur quelle colonne jouer
    while True:
        colonne = safe_int(input('Colonne entre 1 et 7 :'))
        if colonne == 0:                            # Si ce qu'on a tapé n'est pas un chiffre
            print('Erreur de saisie, on quitte !')  # On quitte pour pouvoir arrêter le jeu
            exit()
        if 1 <= colonne <= 7:                       # Le joueur a saisi une colonne valide
            colonne -= 1                            # En programmation, les indexes commencent à 0 et pas à 1, donc on décale de -1
            if grille[0][colonne] == ' ':           # Si la première case de la colonne est jouable
                break                               # On arrête la boucle infinie qui demande où jouer

    # On cherche la position du pion dans la grille
    yok = None
    for y in range(c_hauteur):
        if grille[y][colonne] == ' ':
            yok = y
        else:
            break
    assert yok is not None                          # Pour la forme, car on a déjà vérifié avant que la colonne était jouable

    # Mise à jour et affichage de la grille
    grille[yok][colonne] = get_symbol(joueur_un)
    for y in range(c_hauteur):
        print('| ' + (' | '.join(grille[y])) + ' |')

    # Quelqu'un a-t-il gagné ?
    if haswon(grille):
        print(f'Joueur {get_symbol(joueur_un)} a gagné !')
        exit()
    if isfull(grille):
        print('Ex-aequo : la grille est pleine !')
        exit()

    # Joueur suivant et on reboucle jusqu'à ce qu'on ne puisse plus jouer
    joueur_un = not joueur_un
