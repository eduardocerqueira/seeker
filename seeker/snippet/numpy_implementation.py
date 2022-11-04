#date: 2022-11-04T16:51:52Z
#url: https://api.github.com/gists/e6f6a18fed69cdc49f914dc38631242d
#owner: https://api.github.com/users/Polaris000

def auc_recall_at_k_np(y_true, y_conf):
    """
    Experiment #2:
    --------------
    Compute AUC under the Recall@k curve using numpy's
    functions.

    y_true: A numpy array of expected predictions
    y_conf: A numpy array of the model's confidence
            scores for each datapoint
            
    Returns: AUC-Recall@k (float)
    """

    # if there are no positive targets (good leads),
    # auc becomes invalid
    if y_true.count(1) == 0:
        return np.nan

    conf_df = pd.DataFrame()
    conf_df["conf"] = y_conf
    conf_df["expected"] = y_true
    conf_df.columns = ["conf", "expected"]
    conf_df = conf_df.sort_values("conf", ascending=False)

    ranking = conf_df["expected"].to_numpy()

    recall_at_k = (ranking == 1).cumsum() / (ranking == 1).sum()

    # calculating ideal recall@k
    ideal_recall_at_k = np.minimum(
        np.ones(len(ranking)),
         np.array(list(range(1, len(ranking) + 1)))/ (ranking == 1).sum()
    )

    return np.trapz(recall_at_k) / np.trapz(ideal_recall_at_k)