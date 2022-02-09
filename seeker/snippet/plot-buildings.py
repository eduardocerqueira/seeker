#date: 2022-02-09T17:12:50Z
#url: https://api.github.com/gists/e0e5fd7700b1bd89ea1196ec8fde98cc
#owner: https://api.github.com/users/JEPooleyOS

# Plot
EDGECOLOR = "#FFFFFF00"
FACECOLOR = "#FFFFFF"
BACKGROUND = "#222222"

fig, ax = plt.subplots(facecolor=BACKGROUND)
gs.plot(facecolor=FACECOLOR, edgecolor=EDGECOLOR, ax=ax)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(facecolor=BACKGROUND)
local_buildings_gdf.plot(facecolor=FACECOLOR, edgecolor=EDGECOLOR, ax=ax)
plt.axis('off')
plt.show()
