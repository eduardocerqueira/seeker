#date: 2022-06-03T17:12:43Z
#url: https://api.github.com/gists/1c561b7fa8bb48830e78dced4c9d9c99
#owner: https://api.github.com/users/Babilinx

from kandinsky import *
from ion import keydown
from random import choice


NOIR, GRIS, GRIS_CLAIR, VERT, BLANC, ROUGE, ORANGE = (0, 0, 0), (120, 120, 120), (170, 170, 170), (0, 255, 0), (255, 255, 255), (255, 0, 0), (241, 196, 15)


class Affichage:
    """Permet l'affichage du jeu, du menu principal"""

    # Menu principal du jeu
    def menu_principal(self):
        fill_rect(0, 0, 320, 240, BLANC)
        draw_string("Le Démineur", int(160 - 5 * 11), 8)
        draw_string("[OK] pour démarrer le jeu", int(160 - 5*25), 36, VERT)
        while Jeu.touche_pressee(self) != 4:
        # while not keydown(4):
            pass

    # Affiche la grile du jeu
    def grille(self):
        fill_rect(0, 0, 320, 240, BLANC)
        for i in range(16):
            if i < 11:
                fill_rect(8, 21+20*i, 301, 1, GRIS)
            fill_rect(8+20*i, 21, 1, 200, GRIS)

    # Curseur indiquant la case sue laquelle le joueur se situe
    def curseur(self, xy, c = ORANGE):
        x, y = xy
        fill_rect(8 + 20*int(x), 21 + 20*int(y), 21, 1, c)
        fill_rect(28 + 20*int(x), 21 + 20*int(y), 1, 21, c)
        fill_rect(8 + 20*int(x), 21 + 20*int(y), 1, 21, c)
        fill_rect(8 + 20*int(x), 41 + 20*int(y), 21, 1, c)

    def drapeau(self, x, y):
        fill_rect(17 + 20*x, 26 + 20*y,3,9,c[12])
        fill_rect(17 + 20*x, 36 + 20*y,3,3,c[12])

    def remplir_case(self, x, y, c = GRIS_CLAIR):
        
        pass
        
    def afficher_chiffres(self):
        pass

class Jeu:
    """Exécute tout les calculs permettant de jouer au jeu"""
    
    def __init__(self):
        self.LOCALISATION_JOUEUR = (0, 0)

    # Place aléatoirement des mines sur le plateau
    def creer_mines(self, NOMBRE_DE_MINES, b = []):
        while len(b) < NOMBRE_DE_MINES:
            x, y = choice(range(15)), choice(range(10))
            if (x, y) not in b:
                b.append((x,y))
        self.LOCALISATION_MINES = b

    # Calcule le nombre de mines à proximité des cases
    def marqueurs_mines(self):
        b = 0
        self.MARQUEURS_MINES = {}
        for x in range(16):
            for y in range(10):
                if (x-1, y) in self.LOCALISATION_MINES:
                    b += 1
                if (x-1, y+1) in self.LOCALISATION_MINES:
                    b += 1
                if (x, y+1) in self.LOCALISATION_MINES:
                    b += 1
                if (x+1, y+1) in self.LOCALISATION_MINES:
                    b += 1
                if (x+1, y) in self.LOCALISATION_MINES:
                    b += 1
                if (x+1, y-1) in self.LOCALISATION_MINES:
                    b += 1
                if (x, y-1) in self.LOCALISATION_MINES:
                    b += 1
                if (x-1, y-1) in self.LOCALISATION_MINES:
                    b += 1
                self.MARQUEURS_MINES[x, y] = b
            b = 0

    # Renvoie le code de la touche pressée quand celle-ci est relâchée
    def touche_pressee(self):
        TOUCHES = [0, 1, 2, 3, 4, 5, 6, 12, 17] # gauche, haut, bas, droite, OK, retour, accueil, shift, effacer
        while True:
            for TOUCHE in TOUCHES:
                if keydown(TOUCHE):
                    while keydown(TOUCHE): True
                    return TOUCHE

    def verifier_mine(self):
        if self.LOCALISATION_JOUEUR in self.LOCALISATION_MINES:
            return True
        else:
            return False

    # Contient toutes les actions que le joueur peut faire pour intéragir avec le jeu
    def action_joueur(self):
        x, y = self.LOCALISATION_JOUEUR
        TOUCHE_PRESSEE = self.touche_pressee()
        """
        Fonctionnement :
        Flèches pour se déplacer
        [accueil] pour quitter la partie
        [effacer] pour retirer un drapeau
        [shift] pour passer du mode 'découvrir' à 'signaler'
        [OK] pour 'découvrir' ou 'signaler'
        """
        if TOUCHE_PRESSEE == 0: # gauche
            if x >= 1 and x <= 14: x -= 1
        if TOUCHE_PRESSEE == 2: # haut
            if y >= 0 and y <= 8: y += 1
        if TOUCHE_PRESSEE == 1: # bas
            if y >= 1 and y <= 9: y -= 1
        if TOUCHE_PRESSEE == 3: # droite
            if x >= 0 and x <= 13: x += 1
        if TOUCHE_PRESSEE == 4: # OK
            if self.STATUT_SHIFT:
                # Mettre un drapeau
                pass
            else:
                # Découvrir la case
                self.creuser(self)
                Jeu.remplir_case(self.LOCALISATION_JOUEUR, BLANC)
        if TOUCHE_PRESSEE == 5: # retour
            fill_rect(0, 0, 320, 240, BLANC)
            return break
        if TOUCHE_PRESSEE == 6: # acceuil
            return main()
        if TOUCHE_PRESSEE == 12: # shift
            if self.STATUT_SHIFT != True:
                self.STATUT_SHIFT = True
            else:
                self.STATUT_SHIFT = False
        if TOUCHE_PRESSEE == 17: # effacer
            pass

        Affichage.curseur(self, self.LOCALISATION_JOUEUR, GRIS) # Ici, LOCALISATION_JOUEUR est l'ancienne localisation. Efface l'ancien curseur.
        self.LOCALISATION_JOUEUR = (x, y)
        Affichage.curseur(self, self.LOCALISATION_JOUEUR) # Affiche le nouveau curseur

    # Permet de découvrir une case
    def creuser(self):
        if self.LOCALISATION_JOUEUR in self.LOCALISATION_MINES:
            return False # Retourne False si le joueur marche sur une mine
        else:
            return True # Retourne True si la case ne contient pas de mine

    # Mine automatiquement les cases vides quand on en découvre une
    def minage_auto(self):
        pass


a = Affichage()
j = Jeu()

def main():
    a.menu_principal()
    a.grille()
    j.creer_mines(20)
    j.marqueurs_mines()
    for x in range(16):
        for y in range(10):
            a.remplir_case(x, y)
    a.curseur(j.LOCALISATION_JOUEUR)
    while True:
        j.action_joueur()

main()
