#date: 2022-11-04T16:53:42Z
#url: https://api.github.com/gists/49d01caff4373542ba0eb6fa4facab78
#owner: https://api.github.com/users/Polaris000

@jit
def auc_recall_at_k_np_no_df_numba(y_true, y_conf):
    """
    Experiment #4:
    --------------
    Compute AUC under the Recall@k curve using numpy's
    functions. We do away with the conf_df dataframe 
    as well.

    Numba's jit decorator is also added for further 
    optimization.

    y_true: A numpy array of expected predictions
    y_conf: A numpy array of the model's confidence
            scores for each datapoint
            
    Returns: AUC-Recall@k (float)
    """

    # if there are no positive targets (good leads),
    # auc becomes invalid
    if (y_true == 1).sum() == 0:
        return np.nan

    ranking = y_true[np.argsort(y_conf)[::-1]]

    # calculating recall@k based on sorted ranking
    recall_at_k = (ranking == 1).cumsum() / (ranking == 1).sum()

    # calculating ideal recall@k
    ideal_recall_at_k = np.minimum(
        np.ones(len(ranking)),
         np.array(list(range(1, len(ranking) + 1)))/ (ranking == 1).sum()
    )

    return np.trapz(recall_at_k) / np.trapz(ideal_recall_at_k)