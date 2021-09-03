#date: 2021-09-03T17:06:52Z
#url: https://api.github.com/gists/82c73b6ac6370bfd5ac594e8864b18a0
#owner: https://api.github.com/users/BexTuychiev

print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
    
-----------------------------------------------------
Best value (binary_logloss): 0.35738
	Best params:
		device: gpu
		lambda_l1: 7.71800699380605e-05
		lambda_l2: 4.17890272377219e-06
		bagging_fraction: 0.7000000000000001
		feature_fraction: 0.4
		bagging_freq: 5
		max_depth: 5
		num_leaves: 1007
		min_data_in_leaf: 45
		min_split_gain: 15.703519227860273
		learning_rate: 0.010784015325759629
		n_estimators: 10000