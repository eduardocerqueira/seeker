#date: 2021-09-06T17:01:49Z
#url: https://api.github.com/gists/2203439d36314e15e2daa3f10f79a8df
#owner: https://api.github.com/users/brunobelluomini

import seaborn as sns
from galeritas import bar_plot_with_population_proportion

df = sns.load_dataset("penguins")

bar_plot_with_population_proportion(
    df,
    x='species',
    y='body_mass_g',
    plot_title='Distribuição da massa (g) por espécie',
    figsize=(12, 6)
)