#date: 2023-02-27T16:56:39Z
#url: https://api.github.com/gists/af622d5b7764c6a69524e49803547764
#owner: https://api.github.com/users/giuam2411

# Create multiple radar charts
def make_multiple_radar_charts(name, dataframe, n_clusters, figsize=(18,11), n_wrap=12, savefig=False):
    """
    name: Name of plot (for title)
    dataframe: Pandas DataFrame. Should contain one column named "Clusters" with cluster IDs. 
    n_clusters: No. of clusters
    """
    
    # Get colors
    colors = sns.color_palette("Dark2", n_colors=n_clusters)
    
    # Get the means for each cluster
    dataframe_grouped = dataframe.groupby("Clusters").mean()
    
    # Convert the labels to a numpy array
    labels = np.array(dataframe_grouped.columns)
    labels = ['\n'.join(wrap(label, n_wrap)) + f'\n [{round(dataframe_grouped[label].min(),2)}-{round(dataframe_grouped[label].max(),2)}]' for label in labels] # Add ranges to labels
    #labels = ['\n'.join(wrap(label, n_wrap)) for label in labels] 
    
    # Scale the data from 0 to 100%
    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataframe_grouped)*100 + 5 # Add 5 for nicer representation

    # Get the angles at which a label should be placed
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    
    # Add the first entry to the dimensions to close the line
    angles = np.concatenate((angles,[angles[0]]))
    labels = np.concatenate((labels,[labels[0]]))
    
    # Plot the figure
    fig, axes = plt.subplots(1, n_clusters, figsize=figsize, subplot_kw=dict(polar=True))
    
    for i, ax in enumerate(axes):
        stats = data[i]
        # Add the first entry to the dimensions to close the line
        stats = np.concatenate((stats,[stats[0]]))

        ax.plot(angles, stats, 'o-', color=colors[i], linewidth=2)
        ax.fill(angles, stats, facecolor=colors[i], alpha=0.25)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_yticklabels([])
        ax.set_ylim(bottom=0, top=110)
        ax.set_title(f"Cluster {i}", fontsize=14)
        ax.grid(True)
        
        # Add gap between labels and radar plot and change their orientation
        for label,i in zip(ax.get_xticklabels(),range(0,len(angles))):
            angle_rad=angles[i]
                
            if angle_rad == 0:
                ha= 'left'
                va= "center"
                
            elif angle_rad < np.pi/2:
                ha= 'left'
                va= "bottom"
            
            elif angle_rad == np.pi/2:
                ha= 'center'
                va= "bottom"
            
            elif np.pi/2 < angle_rad < np.pi:
                ha= 'right'
                va= "bottom"

            elif angle_rad == np.pi:
                ha= 'right'
                va= "center"
            
            elif np.pi < angle_rad < (3*np.pi/2):
                ha= 'right'
                va= "top"  
            
            elif angle_rad == (3*np.pi/2):
                ha= 'center'
                va= "top"

            else:
                ha= 'left'
                va= "top"

            label.set_verticalalignment(va)
            label.set_horizontalalignment(ha)


    fig.suptitle(name,y=0.8, fontsize=18)
    fig.tight_layout()
    
    # Save the figure
    if savefig == True: 
        fig.savefig("./images/%s.png" % name)

    return plt.show()
    # Create a large figure with subplots
    #fig, axes = plt.subplots(1, n_clusters, figsize=(10,5))
    
    return plt.show()
