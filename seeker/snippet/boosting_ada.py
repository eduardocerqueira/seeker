#date: 2021-08-31T13:10:56Z
#url: https://api.github.com/gists/0349881e6ccac8107b41adbb0dcc82b5
#owner: https://api.github.com/users/michelkana

from sklearn.ensemble import AdaBoostClassifier

# function to run boosting with gradient descent
def run_adaboosting(X_train, y_train, X_test, y_test, depths=[3], iterations=800, lr=0.05):
    fig, ax = plt.subplots(1,2,figsize=(20,5))
    ab_train_scores = np.zeros((iterations, len(depths)))
    ab_test_scores = np.zeros((iterations, len(depths)))
    for i, depth in enumerate(depths):
        ab_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=iterations, learning_rate=lr)
        ab_model.fit(X_train, y_train);
        train_scores = list(ab_model.staged_score(X_train,y_train))
        test_scores = list(ab_model.staged_score(X_test, y_test))
        ab_train_scores[:, i] = np.array(train_scores)
        ab_test_scores[:, i] = np.array(test_scores)
        ax[0].plot(train_scores,label='depth-{}'.format(depth))
        ax[1].plot(test_scores,label='depth-{}'.format(depth))
    ax[0].set_xlabel('number of iterations', fontsize=12)
    ax[1].set_xlabel('number of iterations', fontsize=12)
    ax[0].set_ylabel('Accuracy', fontsize=12)
    ax[0].set_title("Variation of Accuracy with Iterations (training set)", fontsize=14)
    ax[1].set_title("Variation of Accuracy with Iterations (test set)", fontsize=14)
    ax[0].legend(fontsize=12);
    ax[1].legend(fontsize=12);
    return ab_train_scores, ab_test_scores
  
# run gradient boosting for tree of depth 1, 2, 3, and 4 
tree_depths = [1,2,3,4]
ab_train_scores, ab_test_scores = run_adaboosting(X_train, y_train, X_test, y_test, depths=tree_depths)  
