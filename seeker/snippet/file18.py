#date: 2022-04-15T16:49:48Z
#url: https://api.github.com/gists/2e350c9a16c2246a1768b7c400baa271
#owner: https://api.github.com/users/ShubhashreeSur

alpha = [10**i for i in range(-5,3)]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(alpha=i,loss='hinge',class_weight='balanced')
    clf.fit(X_train, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_train, y_train)
    sig_clf_probs = sig_clf.predict_proba(X_cv)
    cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(y_cv, sig_clf_probs)) 

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.xticks(alpha)
plt.title("Cross Validation Error for each parameter value")
plt.xlabel("Alpha's")
plt.ylabel("Log loss")
plt.show()


best_alpha = np.argmin(cv_log_error_array)
svm_clf = SGDClassifier(alpha=alpha[best_alpha],loss='hinge',class_weight='balanced')
svm_clf.fit(X_train, y_train)
svm_sig_clf = CalibratedClassifierCV(svm_clf, method="sigmoid")
svm_sig_clf.fit(X_train, y_train)


predict_y = svm_sig_clf.predict_proba(X_train)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = svm_sig_clf.predict_proba(X_cv)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = svm_sig_clf.predict_proba(X_test)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
