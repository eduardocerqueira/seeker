#date: 2022-01-12T17:17:48Z
#url: https://api.github.com/gists/a5c41c9ff35a1ecdcd351a787ab3126c
#owner: https://api.github.com/users/haykaza

#initiate logistic regression model
lr = LogisticRegression(solver = 'liblinear', penalty = 'l2', random_state = 0)

#define cross validation parameters
cv = RepeatedStratifiedKFold(n_splits = 20, n_repeats = 5, random_state = 0)

#run cross validation on logistic regression model with defined CV parameters
scoresNoSent = cross_validate(lr, X_NoSent, y, scoring=['roc_auc'], cv = cv, n_jobs = -1)
scoresBertRna = cross_validate(lr, X_BertRna, y, scoring=['roc_auc'], cv = cv, n_jobs = -1)
scoresFinbert = cross_validate(lr, X_Finbert, y, scoring=['roc_auc'], cv = cv, n_jobs = -1)

#report cross validation outputs on 3 datasets
print('\033[1m' + "Model with no sentiment variable" + '\033[0m')
print("AUC:" + str(round(scoresNoSent['test_roc_auc'].mean(),2)))

print('\033[1m' + "With Sentiment from BERT-RNA" + '\033[0m')
print("AUC:" + str(round(scoresBertRna['test_roc_auc'].mean(),2)))

print('\033[1m' + "With Sentiment from FinBert" + '\033[0m')
print("AUC:" + str(round(scoresFinbert['test_roc_auc'].mean(),2)))