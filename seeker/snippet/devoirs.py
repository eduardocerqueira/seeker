#date: 2022-01-03T17:13:48Z
#url: https://api.github.com/gists/a338339c007c81668479092fde25a37a
#owner: https://api.github.com/users/alphadevjs

#tu ouvres le fichier
fichier = open("semaine1_début.txt", "r")
#tu recup toutes les lignes
lignes = fichier.readlines()

#c'est ici que tout le résultat va s'afficher
resultat = ""

#tu prends chaque ligne
for ligne in lignes:
    #tu mets tous les caractères en minuscules
    ligne = ligne.lower()

    #pour chaque caractère dans la ligne
    for letter in ligne.split():
        #tu regardes si c'est un P ou un B
        if letter == "p" or letter == "b":
            #si oui tu remplaces par la nouvelle lettre
            letter = "m"
        
        #tu la lettre au resultat
        resultat += letter
    
    #tu sautes une ligne à la fin de chaque ligne
    resultat += "\n"

fichier.close()

#tu ouvres un nouveau fichier ou tu vas ecrire le résultat
fichier2 = open("semaine1_fin.txt", "w")
fichier2.write(resultat)
#tu fermes les fichier pour enregistrer
fichier2.close()