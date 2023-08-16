#date: 2023-08-16T16:52:16Z
#url: https://api.github.com/gists/1d5127cd0fdd34a05b6978a82cecde65
#owner: https://api.github.com/users/cosmicBboy

from flytekit import task, workflow


@task
def get_data() -> pd.DataFrame:
    return load_dataset()[[TARGET] + FEATURES].dropna()

@task
def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=random_state)

@task
def train_model(
    data: pd.DataFrame,
    hyperparameters: Hyperparameters,
) -> LogisticRegression:
    return LogisticRegression(**asdict(hyperparameters)).fit(data[FEATURES], data[TARGET])

@task
def evaluate(model: LogisticRegression, data: pd.DataFrame) -> float:
    return float(accuracy_score(data[TARGET], model.predict(data[FEATURES])))
