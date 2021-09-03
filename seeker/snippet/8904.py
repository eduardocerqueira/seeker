#date: 2021-09-03T17:06:28Z
#url: https://api.github.com/gists/bdd5e9fad075019ee3dab8b7b690ab6f
#owner: https://api.github.com/users/BexTuychiev

study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=20)