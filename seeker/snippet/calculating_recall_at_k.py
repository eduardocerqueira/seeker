#date: 2022-11-04T16:56:43Z
#url: https://api.github.com/gists/b7e59da65743c1f98b9bcda06338c200
#owner: https://api.github.com/users/Polaris000

    # calculating recall@k
    for i in range(len(conf_df)):
        recall_at_k.append(
            conf_df.iloc[:i+1, :]["expected"].to_list().count(1)
            / conf_df["expected"].to_list().count(1)
        )