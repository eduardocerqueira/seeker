#date: 2021-09-08T17:00:02Z
#url: https://api.github.com/gists/5fd1b444c6b12d985faec73b82cc03cc
#owner: https://api.github.com/users/kamlesh11

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

reg = LogisticRegression(penalty='l2',C=0.01)
min_max=MinMaxScaler()

bef_scaling = reg.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']],Y_train)

X_train_minmax = min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax = min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

reg.fit(X_train_minmax,Y_train)

print("After scaling: - ",accuracy_score(Y_test,bef_scaling.predict(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])))
print("After scaling: - ",accuracy_score(Y_test,reg.predict(X_test_minmax)))
