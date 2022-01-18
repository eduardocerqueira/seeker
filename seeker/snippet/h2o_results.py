#date: 2022-01-18T17:15:39Z
#url: https://api.github.com/gists/3ca4bd34193db2aa7103d4472901eaf7
#owner: https://api.github.com/users/bengchew-lab

import h2o
from  h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML

import pandas as pd
import os


# load data
file = os.getcwd() + "/data/student/student-mat.csv"
math_portuguese = pd.read_csv(file, delimiter=";")
math_portuguese.head()

# initial h2o instance
h2o.init()

# change h2o format
math_portuguese_hf = h2o.H2OFrame(math_portuguese)

# split train and test
train, test = math_portuguese_hf.split_frame(
    ratios = [0.8],
    destination_frames = ['math_portuguese_train', 'math_portuguese_test'],
    seed = 123
)

# print no. of row for train and test df
print("Number of rows for train and test dataset")
print("%d/%d" % (train.nrows, test.nrows))

# filter x and y columns
y = "G3"
ignore_fields = [y]
x = [i for i in train.names if i not in ignore_fields]

# cross validation of models
m_10cv = H2OAutoML(max_runtime_secs = 120, seed = 1, nfolds=10, project_name = "h2o_10folds")
m_10cv.train(x, y, train)

# Show comparison of all models results
m_10cv.leaderboard.head()

# Show performacne for testing dataset 
test_perf = m_10cv.leader.model_performance(test)
print(test_perf)


# output train and test dataset for consistent comparison with modelling method
train_df = train.as_data_frame()
test_df = test.as_data_frame()
train_df.to_csv('../data/train_df.csv')
train_df.to_csv('../data/test_df.csv')


# shutdown h2o instances
h2o.shutdown()