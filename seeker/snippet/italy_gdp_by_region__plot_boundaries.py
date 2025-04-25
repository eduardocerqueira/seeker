#date: 2025-04-25T16:45:04Z
#url: https://api.github.com/gists/26307acf1216397b1f8f28581a12c0fd
#owner: https://api.github.com/users/bianconif

#Create and empty figure
fig, ax = plt.subplots()

#Define the map projection
df_gdata = df_gdata.to_crs('ESRI:53004')

#Plot the boundaries
df_gdata.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)

#Suppress all axis decorations
ax.set_axis_off()