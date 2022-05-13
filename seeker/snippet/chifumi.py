#date: 2022-05-13T17:04:03Z
#url: https://api.github.com/gists/0ced51e4cfdc236189e99a2a2c579266
#owner: https://api.github.com/users/Dan-Lysight

# chifumi.py
# Dan et Alain, 13 mai 2022
# version 1.0 (Python 3.7.9)


import random
T =["pierre", "papier", "ciseaux"]

i_tour = 0 # nombre de tours
i_gain = 0 # nombre de victoires du joueur
i_gaino = 0 # nombre de victoires de l'ordinateur
flag = True # Le jeu continue tant que flag est vrai.
nom_joueur = input("Salut bienvenue dans mon programme veuillez entrer votre nom s'il vous plait : ")
print ("ok "+nom_joueur+",")
print ("")

while (flag):
    i_tour = i_tour+1 # compteur de tour
    computer = T[random.randint(0, 2)] # jeu de l'ordinateur
    joueur = input("Choisissez pierre, papier ou ciseaux (ou stop pour arrêter) : ")

    if computer == "pierre":
        if joueur == "pierre":
            print ("J'avais "+computer+" donc c'est une égalité.")
            print ("")

        elif joueur == "papier":
            print ("J'avais "+computer+" donc c'est toi, "+nom_joueur+", qui gagnes GG.")
            print ("")
            i_gain = i_gain+1
            
        elif joueur == "ciseaux":
            print ("J'avais "+computer+" donc c'est moi qui gagne :).")
            print ("")
            i_gaino = i_gaino+1

        elif joueur == "stop":
            break
                

        else:
            print ("Tu malheusement mal écrit.")
            print ("")
            
    
    elif computer == "papier":
        if joueur == "papier":
            print ("J'avais "+computer+" donc c'est une égalité.")
            print ("")
            
        elif joueur == "ciseaux":
            print ("J'avais "+computer+" donc c'est toi, "+nom_joueur+", qui gagnes GG.")
            print ("")
            i_gain=i_gain+1
            
        elif joueur == "pierre":
            print ("J'avais "+computer+" donc c'est moi qui gagne :).")
            print ("")
            i_gaino = i_gaino+1
            
        elif joueur == "stop":
            break
            
        else:
            print ("Tu as maleureusement mal écrit.")
            print ("")
            

    else:
        if joueur == "ciseaux":
            print ("J'avais "+computer+" donc c'est une égalité.")
            print ("")
            
        elif joueur == "pierre":
            print ("J'avais "+computer+" donc c'est toi, "+nom_joueur+", qui gagnes GG.")
            print ("")
            i_gain = i_gain+1
            
        elif joueur == "papier":
            print ("J'avais "+computer+" donc c'est moi qui gagne :).")
            print ("")
            i_gaino = i_gaino+1
                
        elif joueur == "stop":
            break
            
        else:
            print ("Tu as maleureusement mal écrit.")
            print ("")
            
if i_gain > i_gaino:
    print ("J'ai gagné "+str(i_gaino)+" fois, et vous avez gagné "+str(i_gain)+" fois, donc vous avez gagné.")
elif i_gain < i_gaino:
    print ("J'ai gagné "+str(i_gaino)+" fois, et vous avez gagné "+str(i_gain)+" fois, donc je gagne.")
else:
    print ("J'ai gagné "+str(i_gaino)+" fois, et vous avez gagné "+str(i_gain)+" fois, donc c'est une égalité.")