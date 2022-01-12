#date: 2022-01-12T17:08:52Z
#url: https://api.github.com/gists/384dba04dd9a1863ef94331b6094a8a2
#owner: https://api.github.com/users/haykaza

#apply dimensionality reduction
sentiment_pca = pca.fit_transform(sentiment.to_numpy())

#predict labels for news headlines
sentL = logistic_regression_pca.predict(sentiment_pca)

#add label list to the headlines dataframe
headlines['sentimentLabs'] = sentL
headlines