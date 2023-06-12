#date: 2023-06-12T16:56:34Z
#url: https://api.github.com/gists/6f986267ac66c55c08cf8442c6e67bb9
#owner: https://api.github.com/users/AbSsEnT

# Define preprocessing pipeline.
columns_to_scale = [key for key in FEATURE_TYPES.keys() if FEATURE_TYPES[key] == "numeric"]
columns_to_encode = [key for key in FEATURE_TYPES.keys() if FEATURE_TYPES[key] == "category"]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), columns_to_scale),
    ('cat', OneHotEncoder(handle_unknown='ignore',drop='first'), columns_to_encode)
])


# Train and evaluate model.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=RANDOM_SEED))
])
    
pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print(f'Test Accuracy: {accuracy:.3f}')