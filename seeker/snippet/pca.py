#date: 2023-05-01T16:55:46Z
#url: https://api.github.com/gists/455b205d86ad3829302002d9749e1f57
#owner: https://api.github.com/users/Asmolovskij

pca_1_component = PCA(n_components=1, random_state=RANDOM_STATE)
df_transformed_1_component = pd.DataFrame(data=pca_1_component.fit_transform(df.drop('is_fraud', axis=1)),
                                          columns=['PC1'])
df_transformed_1_component['is_fraud'] = df['is_fraud']

pca_2_component = PCA(n_components=2, random_state=RANDOM_STATE)
df_transformed_2_components = pd.DataFrame(pca_2_component.fit_transform(df.drop('is_fraud', axis=1)),
                                          columns=['PC1', 'PC2'])
df_transformed_2_components['is_fraud'] = df['is_fraud']