#date: 2023-08-09T17:08:18Z
#url: https://api.github.com/gists/e4af87ad1c1c4d1250284b456cc6df8c
#owner: https://api.github.com/users/ygivenx

def bootstrap_ci(data, func, true_col, pred_col, duration_col=None, n_bootstrap=200, alpha=0.05):
    """
    Calculate the confidence interval using bootstrapping.

    Parameters
    ----------
    data: DataFrame
        The data
        
    func: callable
        The function to compute the statistic of interest (e.g. np.mean, np.median)
    true_col: str
        The name of the True column values
    pred_col: str
        The name of the predicted probabilities
    n_bootstrap: int, optional
        The number of bootstrap samples to generate (default: 200)
    alpha: float, optional
        The desired significance level (default: 0.05)

    Returns
    -------
    tuple
        The lower, upper and mean of the confidence interval
    """
    stat = []
    for i in range(n_bootstrap):
        if not duration_col:
            y_true, y_pred = resample(data[true_col],
                                      data[pred_col],
                                      n_samples=len(data),
                                      random_state=i)
            stat.append(func(y_true, y_pred))
        else:
            y_true, y_pred, y_obs = resample(
                data[true_col] == 1,
                data[pred_col],
                data[duration_col],
                n_samples=len(data),
                random_state=i)
            stat.append(func(y_obs, -y_pred, y_true))
            
    lower = np.percentile(stat, 100 * (alpha / 2))
    upper = np.percentile(stat, 100 * (1 - alpha / 2))
    mean = np.mean(stat)
    return round(lower, 2), round(upper, 2), round(mean, 2), stat