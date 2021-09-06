#date: 2021-09-06T17:06:27Z
#url: https://api.github.com/gists/edfb21b2262db98fff296dc3cd171c05
#owner: https://api.github.com/users/brunobelluomini

import seaborn as sns
from galeritas import plot_ecdf_curve

df = sns.load_dataset("penguins")

plot_ecdf_curve(
    df,
    column_to_plot='body_mass_g',
    hue='species',
    figsize=(12, 6),
    plot_title='ECDF da massa corpórea por Ilha'
)