#date: 2021-09-08T17:10:28Z
#url: https://api.github.com/gists/0bcbb63bc1e71334ff7ce90aa988a879
#owner: https://api.github.com/users/kamlesh11

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
log=LogisticRegression(penalty='l2',C=.01)

X_train_scale=scaler.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

X_test_scale=scaler.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

log.fit(X_train_scale,Y_train)

accuracy_score(Y_test,log.predict(X_test_scale))