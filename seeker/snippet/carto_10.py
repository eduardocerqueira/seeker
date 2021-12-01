#date: 2021-12-01T17:01:19Z
#url: https://api.github.com/gists/0f3465a00d0e4b4acca577ee1b801acb
#owner: https://api.github.com/users/Callendar

#Création d'une carte vierge
location = [47, 1]
zoom = 6
tiles = 'cartodbpositron'

Carte = folium.Map(location = location,
                   zoom_start = zoom,
                   tiles = tiles)

#Ajoute du chloroplèthe
chloropleth.add_to(Carte)

#Enregistrement
Carte.save("ensoleillement.html")