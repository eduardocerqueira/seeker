#date: 2021-11-19T16:59:11Z
#url: https://api.github.com/gists/ef3d1d2b73a1c20cb136c345a7b45145
#owner: https://api.github.com/users/Callendar

#Position du marqueur
location = [45.8, 1.2]

#Texte à afficher lorsqu'on clique sur le marqueur
texte = "Limoge est là (à peu près)"

#Création du marqueur
marqueur = folium.Marker(location = location,
                        popup = texte)

#Ajoute à la carte
marqueur.add_to(Carte)

display(Carte)