#date: 2022-04-22T17:00:51Z
#url: https://api.github.com/gists/2841fc1b4bf7bb34deb4d61863d3a7f3
#owner: https://api.github.com/users/shigabeev

from sklearn import metrics

def equal_error_rate(y_pred, y_true):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)
    return eer