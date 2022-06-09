#date: 2022-06-09T16:56:57Z
#url: https://api.github.com/gists/7b9bd5a1b66128f168b9bf17fd90685d
#owner: https://api.github.com/users/arthurcerveira

from xgboost import XGBClassifier

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])     

model_xgb = XGBClassifier(n_jobs=1,scale_pos_weight=scale_pos_weight)

model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)

report = classification_report(y_test, y_pred, target_names=["Não Convertido", "Convertido"])

print(report)

fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

plot_confusion_matrix(model_xgb, X_test, y_test, 
                      ax=ax, cmap=plt.cm.Blues,
                      display_labels=["Não Convertido", "Convertido"])

plt.show()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(model_xgb, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

print(f'Mean ROC AUC: {np.mean(scores):.4f}')

plot_roc_curve(model_xgb, X_test, y_test)