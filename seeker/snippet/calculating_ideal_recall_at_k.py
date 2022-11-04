#date: 2022-11-04T16:57:26Z
#url: https://api.github.com/gists/569dc35a9b3db4c8dc8ed57d24521536
#owner: https://api.github.com/users/Polaris000

# calculating ideal recall@k
ideal_recall_at_k = np.minimum(
    np.ones(len(conf_df)),
    np.array(list(range(1, len(conf_df["expected"]) + 1)))/ conf_df["expected"].to_list().count(1)
)