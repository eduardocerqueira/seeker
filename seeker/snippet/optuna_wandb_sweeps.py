#date: 2022-04-15T16:53:38Z
#url: https://api.github.com/gists/f246c2368be4d1c7592d69cdaeca4d91
#owner: https://api.github.com/users/xadrianzetx

import time

import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
from optuna.integration.wandb import WeightsAndBiasesCallback

import wandb

wandbc = WeightsAndBiasesCallback(metric_name="accuracy", as_multirun=True)


@wandbc.track_in_wandb()
def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    n_estimators = trial.suggest_int("n_estimators", 5, 25)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    classifier_obj = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_split=min_samples_split,
    )

    start = time.time()
    score = sklearn.model_selection.cross_val_score(
        classifier_obj, x, y, n_jobs=-1, cv=3
    )
    elapsed = time.time() - start
    wandb.log({"elapsed": elapsed})
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[wandbc])
    print(study.best_trial)