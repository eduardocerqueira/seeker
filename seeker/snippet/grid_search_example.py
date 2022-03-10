#date: 2022-03-10T16:55:53Z
#url: https://api.github.com/gists/7aa11cae8132b896b14e59b6651c407f
#owner: https://api.github.com/users/kyleziegler

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def grid_search_example():    
    # Parameter grid, grid search will compute all combinations and 
    # scores for the params below.
    pg = {'C': [.005,.01,.05,.1,1,10], 'penalty':['l1','l2']}
    
    model = LogisticRegression(random_state=42, solver='liblinear')
    
    # Here we are optimizing for precision, you can choose from a number of 
    # different metrics to optimize for.
    search = GridSearchCV(estimator=model, param_grid=pg, scoring='precision')
    search.fit(X_train, y_train)
    
    results_df =  pd.DataFrame(search.cv_results_)
    
    results_df = results_df[['param_C', 'param_penalty', 'mean_test_score']]

    l1_l2_score = np.array([results_df[results_df["param_penalty"] == 'l1']['mean_test_score'], 
                       results_df[results_df["param_penalty"] == 'l2']['mean_test_score']])

    return l1_l2_score.T