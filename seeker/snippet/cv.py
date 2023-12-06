#date: 2023-12-06T17:00:05Z
#url: https://api.github.com/gists/88003d7613b140809a57853b24f0af1a
#owner: https://api.github.com/users/hmb2662

# Bull Market
tscv = TimeSeriesSplit(n_splits=10)

param_distributions = {'criterion': ('gini', 'entropy', 'log_loss'),
                       'max_depth': (15,25,35,50),s
                       'max_features': ('sqrt','log2',None),
                       'max_samples': (0.6,0.7,0.8,0.9),
                       'min_samples_leaf': (1, 2, 3, 4),
                       'min_samples_split':(2, 4, 6, 8),
                       'class_weight': ({1:0.25,0:0.5,-1:0.25}, {1:0.25,0:0.45,-1:0.3},None)}

rs = RandomizedSearchCV(estimator=model,
                        param_distributions=param_distributions,
                        n_iter=200,
                        scoring='accuracy',
                        cv=tscv)

rs.fit(xtrain, ytrain)