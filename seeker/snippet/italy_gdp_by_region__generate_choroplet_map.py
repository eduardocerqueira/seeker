#date: 2025-04-25T16:56:55Z
#url: https://api.github.com/gists/cfcfccca8650be0e7c09b52738e514bf
#owner: https://api.github.com/users/bianconif

#Remove the previous plot
ax.clear()

#Generate the choroplet map
df_merged.plot(ax=ax, column=str(year), edgecolor='black', cmap=cpalette, linewidth=0.3)

#Suppress all axis decorations
ax.set_axis_off()

display(fig)