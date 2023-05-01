#date: 2023-05-01T17:00:34Z
#url: https://api.github.com/gists/006495edd63fce60f1f44aa0bf1c4c68
#owner: https://api.github.com/users/Asmolovskij

sns.scatterplot(data=df_transformed_1_component, x='PC1', y=[0]*len(df_transformed_1_component),
               hue='is_fraud')