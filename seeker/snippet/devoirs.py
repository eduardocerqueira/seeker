#date: 2021-10-18T17:01:47Z
#url: https://api.github.com/gists/9c3f0abbdd95e831808ce6ff69d8c674
#owner: https://api.github.com/users/alphadevjs

#Exos
#2.1
def testGrillePleine():
    global continuerJeu;
    plein = True;
    for ligne in grille[0]:
        if ligne == 0:
            plein = False;

    if plein == True:
        print("La grille est pleine");
        continuerJeu = False;

#2.2
def testVictoireLigne (L, numJoueur):
    global continuerJeu;
    compteur = 0;
    victoire = False;
    for i in grille[L]:
        if i == numJoueur:
            compteur += 1;
            if compteur == 4: victoire = True;
        else: i == 0;
    if victoire == True:
        print("Joueur " + str(numJoueur) + " a gagné");
        continuerJeu = False;

#2.3
def testVictoireColonne (C, numJoueur):
    global continuerJeu;
    compteur = 0;
    victoire = False;
    for i in range(6):
        if grille[i][C] == numJoueur:
            compteur += 1;
            if compteur == 4: victoire = True;
        else: i == 0;
    if victoire == True:
        print("Joueur " + str(numJoueur) + " a gagné");
        continuerJeu = False;