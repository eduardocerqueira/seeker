#date: 2022-03-25T17:03:37Z
#url: https://api.github.com/gists/b9929331d08fdccad6034a137e010f74
#owner: https://api.github.com/users/harpreetsahota204

cc_df.drop(columns='CUST_ID', inplace=True)
cc_df.to_csv('cc_df_imputed.csv')
# Since k-means uses Euclidean distance, it would be a good to scale the data
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(cc_df)
save('cc-data-scaled.npy', creditcard_df_scaled)
data_artifacts = {
'cc_df':{'df':'cc_df_imputed.csv',
'type':'data-model',
'alias':['raw-features'],
'metadata':{'filetype':'csv', 'notes':'This dataset contains median imputed values for MINIMUM_PAYMENTS'}
},
'cc_df_scaled':{'df':'cc-data-scaled.npy',
'type':'numpy-array',
'alias':['scaled-features'],
'metadata':{'filetype':'npy', 'notes':'Scaled dataset saved as numpy ndarray.'}
},
}
def artifact_logger(artifact_dict:dict, key: dict, ws:str ,exp_name:str, exp_tag:str):
"""Log the artifact to Comet
Args:
artifact_dict (dict): dictionary containing metadata for artifact
ws(str): Workspace name
key (str): The key from which to grab dictionary items
exp_name (str): Name of the experiment on Comet
exp_tag (str) : Experiment tag
"""
experiment = Experiment(workspace=ws,project_name=exp_name)
experiment.add_tag(exp_tag)
experiment.set_name('log_artifact_' + key)
artifact = Artifact(name = key,
artifact_type = artifact_dict[key]['type'],
aliases = artifact_dict[key]['alias'],
metadata = artifact_dict[key]['metadata']
)
artifact.add(artifact_dict[key]['df'])
experiment.log_artifact(artifact)
experiment.end()
# Log training and testing sets to Comet as artifacts
for key in data_artifacts:
artifact_logger(data_artifacts,key, ws='team-comet-ml', exp_name='cc-clustering', exp_tag="imputed-data")