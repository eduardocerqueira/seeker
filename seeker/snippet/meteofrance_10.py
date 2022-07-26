#date: 2022-07-26T16:55:24Z
#url: https://api.github.com/gists/9480731a3f146c437c08a5697d668301
#owner: https://api.github.com/users/Callendar

import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

#Ouverture de la projection
with xr.open_rasterio('temp.tiff') as temp:
    da_to_plot = temp.sel(band = 1, drop = True)

#Coordonnées de la zone à représenter
lat_min, lat_max = 40, 52
lon_min, lon_max = -6, 10
central_longitude = (lon_min + lon_max) / 2
central_latitude = (lat_min + lat_max) / 2

#Echelle de température
cmap = "RdBu_r"
milieu = 15
delta = 30

#Initialisation de la figure    
fig = plt.figure(figsize=(12,10), facecolor = 'w')
ax = plt.axes(projection = ccrs.LambertConformal(central_longitude = central_longitude,
                                                 central_latitude = central_latitude))
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

#Représentation de la prévision
g = da_to_plot.plot(transform = ccrs.PlateCarree(),
                    cmap= cmap,
                    vmin = milieu - delta,
                    vmax = milieu + delta,
                    add_colorbar = False,
                    alpha = 0.8)

#Représentation des isothermes
isothermes = [i for i in range (milieu - delta, milieu + delta + 5, 5)]
g2 = da_to_plot.plot.contour(transform = ccrs.PlateCarree(),
                cmap = cmap,
                add_colorbar = False,
                levels = isothermes,
                linewidths = 3)

#Création de l'échelle
cb = plt.colorbar(g,
                  label = "Température (°C)",
                  shrink = 0.7,
                  extend = 'both')
cb.add_lines(g2)

#Affichage des côtes, frontières et océans
ax.coastlines(zorder = 101)
ax.add_feature(cfeature.BORDERS, zorder = 101, color = 'black', ls = ':')
ax.add_feature(cfeature.OCEAN, zorder = 99, color = 'lightgray')

#Titre
ax.set_title("Prévision de température pour le 30 juillet 2022 à 16h",
            fontsize = "xx-large")

plt.show()