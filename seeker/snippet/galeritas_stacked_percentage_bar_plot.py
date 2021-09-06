#date: 2021-09-06T17:09:48Z
#url: https://api.github.com/gists/affb04b8f87b1cf711567bb93aae114d
#owner: https://api.github.com/users/brunobelluomini

import seaborn as sns
from galeritas import stacked_percentage_bar_plot

df = sns.load_dataset("penguins")

stacked_percentage_bar_plot(
    df,
    categorical_feature='island',
    hue='species',
    annotate=True,
    plot_title='Proporção das espécies de pinguins por ilha',
    figsize=(12, 6)
)