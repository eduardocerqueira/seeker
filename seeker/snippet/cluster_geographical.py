#date: 2022-01-25T17:10:40Z
#url: https://api.github.com/gists/79af1a0e5c274f498265aacdd3eddd5c
#owner: https://api.github.com/users/rvt123

kmeans = KMeans(init="random",n_clusters=16,random_state=42,) # kmeans object
kmeans.fit(data_transformed) # fitting on data
print(kmeans.inertia_)
country_df['CLUSTERS'] = kmeans.labels_ # adding calculated labels to the dataframe

# colour to be used
color_list = ['blue','green','red','cyan','tan','lightcoral','yellow','lightgreen','lime','brown',\
              'coral','darkcyan','khaki','powderblue','darksalmon','orange','lightcyan','plum']
clusters = np.unique(country_df['CLUSTERS']) # list containing unique value of clusters formed

## Legend for the plot using Patches from matplotlib.pyplot
patches = [ eval('''mpatches.Patch(color='{colour}', label='Cluster {cluster_num}')'''\
                         .format(colour=color_list[cluster_num],cluster_num=cluster_num)) for cluster_num in clusters ]

ax2 = world.plot(figsize=(15,15), edgecolor=u'white', color='gray') # defining figure size and default colour of any country (gray)
for cluster_num in clusters: # plotting each cluster in the list one by one
    world[world['NAME'].isin(list(country_df[country_df['CLUSTERS'] == cluster_num]['COUNTRY']))]\
                                                        .plot(edgecolor=u'white', color=color_list[cluster_num], ax=ax2)
    ax2.axis('scaled')

plt.legend(handles=patches,bbox_to_anchor=(1.08, 1)) # Plotting legend for the clusters
# plt.savefig('Home_Page_1.jpg') # if there is need to save the cluster
plt.show() # show the plot