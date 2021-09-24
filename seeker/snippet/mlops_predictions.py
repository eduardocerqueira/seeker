#date: 2021-09-24T17:10:57Z
#url: https://api.github.com/gists/214a491876fea1a6eaab5a4e68d47ba8
#owner: https://api.github.com/users/mtanco

import json

import numpy as np
import pandas as pd
import requests

"""
Suggestions on how to use this in an H2O Wave app:
1. Have a page with a textbox asking users for a URL of a REST endpoint
2. Allow users to upload a CSV with new data
3. Show the predictions as a table
4. Show the distribution of the predictions as a bar chart by decile 
5. Allow a user to choose a categorical feature and see predictions by category
"""


def mlops_get_score(score_url, query):
    """
    Send our data to a REST endpoint and get back predictions
    
    :param score_url: REST endpoint of the model running in H2O MLOps
    :param query: Data we want to get predictions on, formatted as a dictionary
    :return: Predictions from H2O MLOps
    """
    query_json = json.loads(query)
    response = requests.post(url=score_url, json=query_json)
    
    if response.status_code == 200:  # confirm that our model is up and running, we have permissions to it, etc.
        return json.loads(response.text)
    else:
        return None


def df_get_preds_from_mlops(url: str, df: pd.DataFrame):
    """
    Get predictions on data from a model in MLOps, starts and ends as a pandas dataframe

    :param url: REST endpoint of the model running in H2O MLOps
    :param df: Data we want to get predictions on, formatted as a dictionary
    :return: Original data with new columns of predictions
    """
    df.reset_index(drop=True, inplace=True)  # ensure our index starts at 0 for joining on predictions
    rows = df.where(pd.notnull(df), "")  # replace nulls with "" as MLOps expected format

    # Format all values as strings, as MLOps expected format
    values = rows.values.tolist()
    for i in range(len(values)):
        values[i] = [str(x) for x in values[i]]

    # Prepare for json format required by MLOps
    dictionary = (
        '{"fields": ' + str(rows.columns.tolist()) + ', "rows": ' + str(values) + "}"
    )
    dictionary = dictionary.replace("'", '"')

    # Get predictions from MLOps
    dict_preds = mlops_get_score(url, dictionary)

    # Format predictions as a 0 indexed dataframe to join back with features in appropriate datatype
    preds = pd.DataFrame(data=dict_preds["score"], columns=dict_preds["fields"])
    preds = preds.apply(pd.to_numeric)

    predictions_with_features = pd.concat([rows, preds], axis=1)

    # replace space with nulls
    return predictions_with_features.replace(r"^\s*$", np.nan, regex=True)


n = 100
df = pd.DataFrame(dict(
    length=np.random.rand(n),
    width=np.random.rand(n),
    data_type=np.random.choice(a=['Train', 'Test'], size=n, p=[0.8, 0.2])
))


predictions = df_get_preds_from_mlops(
    url="https://model.demo.h2o.ai/9989247c-38b3-4596-ac47-d1f6a0ae5dad/model/score",
    rows=df
)

print(predictions)
