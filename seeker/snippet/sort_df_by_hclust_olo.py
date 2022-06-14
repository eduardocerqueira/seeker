#date: 2022-06-14T17:07:16Z
#url: https://api.github.com/gists/014dec095100ff4e17ce130414c70c52
#owner: https://api.github.com/users/alexlenail

import scipy
import scipy.stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy

def sort_df_by_hclust_olo(df, how='both', method='ward', metric='euclidean'):
    '''
    how={'index', 'columns', 'both'}
    '''
    df = df.fillna(0)

    if how in ['index', 'both']:
        Z = linkage(df, method=method, metric=metric)
        order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, df))
        df = df.iloc[order]

    if how in ['columns', 'both']:
        df = df.T
        Z = linkage(df, method=method, metric=metric)
        order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, df))
        df = df.iloc[order].T

    return df.replace(0, np.nan)
