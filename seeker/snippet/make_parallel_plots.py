#date: 2023-02-27T16:59:24Z
#url: https://api.github.com/gists/2d02a5133a348dea80978fc40e438ce5
#owner: https://api.github.com/users/giuam2411

def parallel_plot_clusters(cols, data, n_clusters):
    """
    cols: labels of columns
    data: Pandas DataFrame. Should contain one column named "Clusters" with cluster IDs. 
    n_clusters: No. of clusters
    """
    
    sns.set_style('white')

    cols = ['\n'.join(wrap(x, 16)) for x in cols]

    ynames = cols
    ys = np.array(data)
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05

    ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=(15,8))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=12)
    host.tick_params(axis='x', which='major', rotation=90, pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title('Parallel Coordinates Plot', fontsize=20, pad=12)

    colors = plt.cm.Dark2.colors
    legend_handles = [None for _ in data["Clusters"]]
    for j in range(ys.shape[0]):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1.5, alpha=0.4, edgecolor=colors[data["Clusters"][j]])
        legend_handles[data["Clusters"][j]] = patch
        host.add_patch(patch)
    host.legend(legend_handles, list(range(0,n_clusters)),
                loc='lower center', bbox_to_anchor=(0.5, -0.18),
                ncol=n_clusters, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()