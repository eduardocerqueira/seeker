#date: 2022-01-05T17:04:53Z
#url: https://api.github.com/gists/140bc0f4906eaee49623e047205216f7
#owner: https://api.github.com/users/yashprakash13

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print(f"Optimized RMSE: {study.best_value:.4f}")
print("Best params:")
for key, value in study.best_params.items():
    print(f"\t{key}: {value}")