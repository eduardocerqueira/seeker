#date: 2022-11-04T16:58:19Z
#url: https://api.github.com/gists/4ce9f4fbbf7c28c983a56295909f6292
#owner: https://api.github.com/users/Polaris000

# calculating ideal recall@k
ideal_recall_at_k = np.minimum(
    np.ones(len(ranking)),
     np.array(list(range(1, len(ranking) + 1)))/ (ranking == 1).sum()
)