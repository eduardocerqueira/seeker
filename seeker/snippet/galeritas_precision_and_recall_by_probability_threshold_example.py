#date: 2021-09-06T17:12:51Z
#url: https://api.github.com/gists/628e09f5f8f0c94f2474da8648b01f92
#owner: https://api.github.com/users/brunobelluomini

import pandas as pd
from galeritas import precision_and_recall_by_probability_threshold

titanic_df = pd.read_csv("tests/data/titanic.csv")

precision_and_recall_by_probability_threshold(
    df=titanic_df,
    prediction_column_name='predict_proba',
    target_name='survived',
    target=1,
    plot_title='Precision, Recall and Support Ratios for Positive Class (survived)',
    figsize=(12, 4)
)