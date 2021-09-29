#date: 2021-09-29T17:03:44Z
#url: https://api.github.com/gists/482c22618c8081c357624e3084a6db38
#owner: https://api.github.com/users/antoineeudes

import mlflow

with mlflow.start_run(): # start an experiment
    mlflow.log_param("layers", layers)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("lr", lr)

    # train model

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("acc", acc)
    mlflow.log_artifact("plot", model.plot(test_df))
    mlflow.tensorflow.log_model(model)
