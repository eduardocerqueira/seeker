#date: 2024-06-27T17:01:39Z
#url: https://api.github.com/gists/0570d69078259ff4fd0c07967a4064ed
#owner: https://api.github.com/users/Math170

from random import *
def PPC():
    #génère un nombre aléatoire entre 1et 3
    choix_robot = randint(1 , 3)
    #transforme le nombre aléatoire en pierre, papier ou ciseaux
    if choix_robot == 1 :
        choix_robot = "pierre"
    elif choix_robot == 2 :
        choix_robot = "papier"
    elif choix_robot == 3 :
        choix_robot = "ciseaux"
    
    #demande au joueur de choisir entre pierre, papier ou ciseaux
    choix_joueur = str(input("Choissisez entre pierre papier ciseaux : "))
    #compare le choix du robot et du joueur pour dire le gagnant 
    if choix_joueur == "pierre" :
        if choix_robot == "pierre" :
            print("Le Robot a fait pierre.")
        elif choix_robot == "papier" :
            print("Perdu, le Robot a fait papier.")
        elif choix_robot == "ciseaux" :
            print("Gagné, le robot a fait ciseaux.")
    elif choix_joueur == "papier" :
        if choix_robot == "pierre" :
            print("Gagné, le robot a fait pierre.")
        elif choix_robot == "papier" :
            print("Le Robot a fait papier.")
        elif choix_robot == "ciseaux" :
            print("Perdu, le Robot a fait ciseaux.")
    elif choix_joueur == "ciseaux" :
        if choix_robot == "pierre" :
            print("Perdu, le Robot a fait pierre.")
        elif choix_robot == "papier" :
            print("Gagné, le robot a fait papier.")
        elif choix_robot == "ciseaux" :
            print("Le Robot a fait ciseaux.")
    elif choix_joueur != "pierre" or "papier" or "ciseaux" :
        print("erreur") #si le joueur entre autre chose que pierre, papier ou ciseaux le message erreur sera envoyé

PPC()