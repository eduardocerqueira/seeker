#date: 2022-03-18T17:02:43Z
#url: https://api.github.com/gists/9c7a79b5f1209a1799b43fb13bcc1704
#owner: https://api.github.com/users/kyleziegler

from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = [10,4]

def grid_search():    
    model = GradientBoostingClassifier(random_state=0, n_estimators=50)
    
    # Param n_iter: Number of parameter settings that are sampled; 
    # tradeoff between runtime vs quality of the solution.
    # Param cv: Number of cross validation folds, default is 3.
    
    opt = GridSearchCV(
        model,
        {
            'learning_rate':list(np.arange(0.1,0.4,0.1)),
            'n_estimators': list(range(50,200,50)),
            'min_samples_split': list(range(2,10,2))
        },
        n_iter=30,
        cv=5,
        n_jobs=100,
        scoring='precision',
        random_state = 0
    )
    
    opt.fit(X_train, y_train)

    x = pd.DataFrame(opt.cv_results_)['mean_test_score'].index
    y = pd.DataFrame(opt.cv_results_)['mean_test_score']
    plot(x,y)

    print("Best Validation Score:", opt.best_score_)
    print("Test Score:", model.score(X_test, y_test))

grid_search()