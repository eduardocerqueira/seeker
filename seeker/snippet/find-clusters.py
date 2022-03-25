#date: 2022-03-25T17:14:24Z
#url: https://api.github.com/gists/d1d8cf5874bd32245e6d63e8df096c79
#owner: https://api.github.com/users/harpreetsahota204

def find_clusters(df:pd.DataFrame, file:str):
"""
Run an experiment to find 3, 4, and 5 clusters.
Parameters:
df: The dataframe on which clustering will take place
file: A string to help add tags, and identifying information for the experiment
"""
for k in range(3,6,1):
file_string = file + "_" + str(k)
experiment = Experiment(workspace='team-comet-ml', project_name='cc-clustering')
experiment.add_tag(file + "_" + str(k) + "_clusters")

kmeans = KMeans(k, random_state=42, algorithm='elkan', n_init = 100)
pickle.dump(kmeans, open(file_string + ".pkl", "wb"))
kmeans.fit(df)
labels = kmeans.labels_
clusters = pd.DataFrame(labels, columns = ["cluster_label"])
cc_df_clusters=pd.concat([cc_df, clusters], axis=1)
cc_df_clusters.to_csv(f'cc_df_{k}_clusters.csv')
score = silhouette_score(df, labels, metric='euclidean')
metrics = {"silhouette_score": score, "inertia": kmeans.inertia_}

experiment.log_model(file_string, file_string + ".pkl")
experiment.log_parameters(k)
experiment.log_metrics(metrics)
experiment.log_table(f'cc_df_{k}_clusters.csv', tabular_data=True, headers=True)
experiment.end()