#date: 2021-12-29T16:58:59Z
#url: https://api.github.com/gists/163e4e07c5eaacabd4d112befc18b476
#owner: https://api.github.com/users/khuyentran1401

import pandas as pd

df = pd.read_csv(
    "https://media.githubusercontent.com/media/khuyentran1401/Data-science"
    "/master/statistics/bayes_linear_regression/student-score.csv"
)

df.head(10)