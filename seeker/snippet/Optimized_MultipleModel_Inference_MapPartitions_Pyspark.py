#date: 2023-09-05T17:03:50Z
#url: https://api.github.com/gists/bb959d199085e9e2590083f6d33f6ff5
#owner: https://api.github.com/users/dineshdharme

This need not be so complicated. Here's an concrete example of how you can do this.

First load all the models in a list. Broadcast the list. Access the list broadcasted variable's value using `value`. You can concatenate your features into an array as I have done below then do inference on samples one by one.
You could achieve the batch semantics by using mapPartition function on an rdd and then convert the result back to dataframe as shown below.





    import sys
    from pyspark import SparkContext, SQLContext
    import joblib
    from pyspark.sql.functions import pandas_udf
    import pandas as pd
    from pyspark.sql import functions as F
    from sklearn.datasets import load_iris
    from keras.layers import Dense
    from keras.models import Sequential
    import numpy as np
    
    sc = SparkContext('local')
    sqlContext = SQLContext(sc)
    
    
    ## Get the data_iris.csv from this location : https://github.com/dehaoterryzhang/Iris_Classification/tree/master/data
    input_file = "../data/data_iris.csv"
    
    initial_df = sqlContext.read.option("inferSchema", "true").csv(input_file, header=True)
    initial_df.show(n=10, truncate=False)
    
    X_train_df = initial_df.select(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y_train_df = initial_df.select("species")
    
    print("X_train_df features info")
    X_train_df.show(n=3, truncate=False)
    print("y_train_df label info")
    y_train_df.show(n=3, truncate=False)
    
    print("distinct classes in the dataset")
    y_train_df.distinct().show(n=100, truncate=False)
    
    all_columns = X_train_df.columns
    X_train_df = X_train_df.withColumn("features_concat", F.array(all_columns))
    
    all_columns_afterarray = X_train_df.columns
    
    X, y = load_iris(return_X_y=True)
    
    model = Sequential([
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)])
    
    model.compile(loss='mean_absolute_error', optimizer='adam')
    
    model.fit(X, y, epochs=10, verbose=0)
    
    keras_model_export_path = '../model_exported/keras_nn_iris_model.pkl'
    ## Uncomment below to export the trained model
    # joblib.dump(model, keras_model_export_path, compress=9)
    
    
    keras_model_loaded_from_path = joblib.load(keras_model_export_path)
    model_clone_list = [ keras_model_loaded_from_path] * 10  ## just to simulate list of 10 models, the models are just copies one model
    
    ## Broadcasting the models so that they are available at the executor
    broadcasted_model_clone_list = sc.broadcast(model_clone_list)
    
    
    def different_inference(features_array):
        X = pd.DataFrame([features_array])
        X.columns = all_columns
    
        prediction_classification_scores_list = []
        # inference over all 10 models in a for loop
        for model_ii in broadcasted_model_clone_list.value:
            curr_result = model_ii.predict(X)
            prediction_classification_scores_list.append(curr_result[:, 0])
    
        prob_scores = np.array(prediction_classification_scores_list)
        final_result = pd.Series(prob_scores.mean(axis=0)).tolist()
    
        return final_result
    
    
    different_inference_udf = F.udf(different_inference)
    
    ### Inferencing over only 10 rows
    #predicted_df = X_train_df.limit(10).withColumn("prediction_scores", different_inference_udf(F.col("features_concat")))
    
    print("Inference result from the 10 models")
    #predicted_df.show(n=10, truncate=False)
    
    
    
    def mapPartition_inference(partitioned_rows):
    
        features_array_list = []
        for row in partitioned_rows:
            features_array_list.append(row.features_concat)
    
        X = pd.DataFrame(features_array_list)
        X.columns = all_columns
    
        prediction_classification_scores_list = []
        # inference over all 10 models in a for loop
        for model_ii in broadcasted_model_clone_list.value:
            curr_result = model_ii.predict(X)
            prediction_classification_scores_list.append(curr_result[:, 0])
    
        prob_scores = np.array(prediction_classification_scores_list)
        final_result = pd.Series(prob_scores.mean(axis=0)).tolist()
        print("Hooray are we here!!!!!!!!!!!!!")
    
        constructed_result = []
        for jj in range(len(features_array_list)):
            constructed_result.append([features_array_list[jj], final_result[jj]])
    
        return iter(constructed_result)
    
    
    partitioned_df = X_train_df.limit(30).repartition(10)
    
    partition_predicted_df = partitioned_df.rdd.mapPartitions(mapPartition_inference).toDF(["features_concat", "prediction_avg_scores"])
    
    print("Inference result from the 10 models using mapPartitions (optimized version)")
    partition_predicted_df.show(n=30, truncate=False)


Output is as follows : 

    +------------+-----------+------------+-----------+-------+
    |sepal_length|sepal_width|petal_length|petal_width|species|
    +------------+-----------+------------+-----------+-------+
    |5.1         |3.5        |1.4         |0.2        |setosa |
    |4.9         |3.0        |1.4         |0.2        |setosa |
    |4.7         |3.2        |1.3         |0.2        |setosa |
    |4.6         |3.1        |1.5         |0.2        |setosa |
    |5.0         |3.6        |1.4         |0.2        |setosa |
    |5.4         |3.9        |1.7         |0.4        |setosa |
    |4.6         |3.4        |1.4         |0.3        |setosa |
    |5.0         |3.4        |1.5         |0.2        |setosa |
    |4.4         |2.9        |1.4         |0.2        |setosa |
    |4.9         |3.1        |1.5         |0.1        |setosa |
    +------------+-----------+------------+-----------+-------+
    only showing top 10 rows
    
    X_train_df features info
    +------------+-----------+------------+-----------+
    |sepal_length|sepal_width|petal_length|petal_width|
    +------------+-----------+------------+-----------+
    |5.1         |3.5        |1.4         |0.2        |
    |4.9         |3.0        |1.4         |0.2        |
    |4.7         |3.2        |1.3         |0.2        |
    +------------+-----------+------------+-----------+
    only showing top 3 rows
    
    y_train_df label info
    +-------+
    |species|
    +-------+
    |setosa |
    |setosa |
    |setosa |
    +-------+
    only showing top 3 rows
    
    distinct classes in the dataset
    +----------+
    |species   |
    +----------+
    |virginica |
    |versicolor|
    |setosa    |
    +----------+
    
    Inference result from the 10 models using mapPartitions (optimized version)
    
    +--------------------+---------------------+
    |features_concat     |prediction_avg_scores|
    +--------------------+---------------------+
    |[5.2, 3.4, 1.4, 0.2]|0.6321244239807129   |
    |[5.1, 3.8, 1.5, 0.3]|0.6290395855903625   |
    |[4.3, 3.0, 1.1, 0.1]|0.5323924422264099   |
    |[4.8, 3.4, 1.6, 0.2]|0.6239675879478455   |
    |[4.9, 3.1, 1.5, 0.1]|0.6233397722244263   |
    |[5.1, 3.5, 1.4, 0.3]|0.625523030757904    |
    |[4.4, 2.9, 1.4, 0.2]|0.5821532607078552   |
    |[5.0, 3.4, 1.5, 0.2]|0.627651035785675    |
    |[4.6, 3.4, 1.4, 0.3]|0.5886028409004211   |
    |[4.8, 3.0, 1.4, 0.1]|0.6058898568153381   |
    |[5.4, 3.9, 1.3, 0.4]|0.6334523558616638   |
    |[5.1, 3.5, 1.4, 0.2]|0.6212030053138733   |
    |[4.6, 3.6, 1.0, 0.2]|0.5346620678901672   |
    |[4.6, 3.1, 1.5, 0.2]|0.6045092344284058   |
    |[4.8, 3.4, 1.9, 0.2]|0.6606692671775818   |
    |[5.0, 3.4, 1.6, 0.4]|0.6499260663986206   |
    |[4.9, 3.0, 1.4, 0.2]|0.618742823600769    |
    |[5.1, 3.7, 1.5, 0.4]|0.6369576454162598   |
    |[5.4, 3.7, 1.5, 0.2]|0.6512832045555115   |
    |[5.2, 3.5, 1.5, 0.2]|0.640670657157898    |
    |[5.8, 4.0, 1.2, 0.2]|0.6450895667076111   |
    |[5.7, 4.4, 1.5, 0.4]|0.6659770607948303   |
    |[5.0, 3.6, 1.4, 0.2]|0.6101197004318237   |
    |[4.7, 3.2, 1.3, 0.2]|0.5853733420372009   |
    |[5.0, 3.0, 1.6, 0.2]|0.6513381004333496   |
    |[5.4, 3.9, 1.7, 0.4]|0.679506242275238    |
    |[4.7, 3.2, 1.6, 0.2]|0.6218041181564331   |
    |[5.7, 3.8, 1.7, 0.3]|0.7012595534324646   |
    |[5.4, 3.4, 1.7, 0.2]|0.6839534640312195   |
    |[5.1, 3.3, 1.7, 0.5]|0.6780367493629456   |
    +--------------------+---------------------+





