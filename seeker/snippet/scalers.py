#date: 2021-09-16T17:14:42Z
#url: https://api.github.com/gists/582dea2ba873af644f2fdca4e996719b
#owner: https://api.github.com/users/sengstacken

scale_cols = ['Time','Amount']
scaler = StandardScaler()

# fit scaler
scaler.fit(train_df[scale_cols].to_numpy())

# make copies of dataframes
train_df_ = train_df.copy()
val_df_ = val_df.copy()
test_df_ = test_df.copy()

# apply scaler
train_df_.loc[:,scale_cols] = scaler.transform(train_df[scale_cols].values)
val_df_.loc[:,scale_cols] = scaler.transform(val_df[scale_cols].values)
test_df_.loc[:,scale_cols] = scaler.transform(test_df[scale_cols].values)