#date: 2023-06-12T17:00:03Z
#url: https://api.github.com/gists/467a72fd1d66e375880e256a59a08962
#owner: https://api.github.com/users/AbSsEnT

from giskard import Model


# Wrap model with Giskard.
wrapped_model = Model(pipeline,
                      model_type="classification",
                      name="Churn classification",
                      feature_names=FEATURE_TYPES.keys())

# Validate wrapped model.
wrapped_Y_pred = wrapped_model.predict(wrapped_data).prediction
wrapped_accuracy = accuracy_score(Y_test, wrapped_Y_pred)

print(f'Wrapped Test Accuracy: {wrapped_accuracy:.3f}')
