#date: 2021-11-09T17:15:34Z
#url: https://api.github.com/gists/8df96e08087e7b42922aeef19df79cff
#owner: https://api.github.com/users/Steboss89

import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow_sdk import experiment_tracking_training

# Define mlflow params
tracking_params ={
    'tracking_uri':'http://127.0.0.1:5000', # use your local aaddress or a server one
    'tracking_storage':'mlruns',  # we're going to save locally these results otherwise use a cloud_storage
    'run_name': 'random_forest_iris', # the name of the experiment's run
    'experiment_name':'random_forest',  # the name of the experiment
    'tags':None,
}

# load the iris dataset
iris = datasets.load_iris()
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

# divide features and labels
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
# Random Forest
clf=RandomForestClassifier(n_estimators=2)
# input parameters for the model
params={'X': X_train,
        'y': y_train}
# I want to save all the artefacts locally as well
output_folder = os.getcwd() + '/outputs'
# retrieve the test labels
test_labels = y_test.unique()
# run the training and fit
run_id = experiment_tracking_training.start_training_job(experiment_tracking_params=tracking_params)
clf.fit(**params)
# end the training
experiment_tracking_training.end_training_job(experiment_tracking_params=tracking_params)
# as a test case add fake metrics
false_metrics = {"test_metric1":0.98,
                 "test_metric2":0.00,
                 "test_metric3":50}
experiment_tracking_training.add_metrics_to_run(run_id, tracking_params, false_metrics)