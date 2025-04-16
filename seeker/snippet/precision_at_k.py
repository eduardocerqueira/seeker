#date: 2025-04-16T16:52:51Z
#url: https://api.github.com/gists/36b4a5337872738d965cf03377bfd648
#owner: https://api.github.com/users/brianhill11

def precision_at_k(y_true: np.array, y_pred: np.array, k: int = 10) -> np.float64:
    """Calculate precision at k for a given model

    Args:
        y_true (np.array): array of true labels
        y_pred (np.array): array of predicted probabilities
        k (int, optional): number of predictions to consider. Defaults to 10.

    Returns:
        np.float64: precision at k
    """
    # convert y_true and y_pred to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values.astype(int)
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    # sort the predictions by probability (descending order)
    y_pred = np.argsort(y_pred)[::-1]
    # get the top k predictions
    y_pred = y_pred[:k]
    # get the precision at k 
    # (and handle case where k is greater than the number of true positives)
    return np.sum(y_true[y_pred]) / min(k, np.sum(y_true))