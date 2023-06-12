#date: 2023-06-12T16:58:38Z
#url: https://api.github.com/gists/a537a27d4bb317fd5bde8571e90ba08d
#owner: https://api.github.com/users/AbSsEnT

from giskard import Dataset


# Prepare data to wrap.
raw_data = pd.concat([X_test, Y_test], axis=1)

# Wrap data with Giskard.
wrapped_data = Dataset(raw_data,
                       name="Churn classification dataset",
                       target=TARGET_COLUMN_NAME,
                       column_types=FEATURE_TYPES)
