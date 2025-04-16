#date: 2025-04-16T16:37:40Z
#url: https://api.github.com/gists/a6821d3f6bd71d2133ff8731bf01c515
#owner: https://api.github.com/users/tankist52

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Логистическая регрессия
log_reg = LogisticRegression(C=0.1, max_iter=50)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print('Логистическая регрессия')
print(classification_report(y_test, y_pred_log, target_names=le.classes_))

# Случайный лес
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('Случайный лес')
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# Градиентный бустинг
xg = GradientBoostingClassifier(n_estimators=100, random_state=42)
xg.fit(X_train, y_train)
xg_pred = xg.predict(X_test)
print('Градиентный бустинг')
print(classification_report(y_test, xg_pred, target_names=le.classes_))


import joblib
joblib.dump(rf, 'Risk_model.pkl')