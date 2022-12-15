#date: 2022-12-15T16:37:30Z
#url: https://api.github.com/gists/0d36b51a832607d32f92155fe72a6882
#owner: https://api.github.com/users/UNDERBI

from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
features_train, features_test, target_train, target_test = train_test_spli
t(vals_sc,target,test_size=0.1,random_state = 0)
kn_model = KNeighborsClassifier(n_neighbors = 10)
rfc_model = RandomForestClassifier (random_state = 0)
ada_model = AdaBoostClassifier(random_state = 0)
log_model = LogisticRegression()
tree_model = DecisionTreeClassifier()
kn_model.fit(features_train, target_train)
ada_model.fit(features_train, target_train)
log_model.fit(features_train, target_train)
tree_model.fit(features_train, target_train)
rfc_model.fit(features_train, target_train)
kn_predict = kn_model.predict(features_test)
rfc_predict = rfc_model.predict(features_test)
ada_predict = ada_model.predict(features_test)
log_predict = log_model.predict(features_test)
tree_predict = tree_model.predict(features_test)
print('KNeighbors', accuracy_score(kn_predict,target_test))
print('RandomForest', accuracy_score(rfc_predict,target_test))
print('AdaBoost', accuracy_score(ada_predict,target_test))
print('LogisticRegression', accuracy_score(log_predict,target_test))
print('DecisionTreeClassifier', accuracy_score(tree_predict,target_test))