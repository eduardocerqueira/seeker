#date: 2023-06-12T16:53:53Z
#url: https://api.github.com/gists/2a1ea81e566dfbc585d8ada5d07c865d
#owner: https://api.github.com/users/AbSsEnT

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns='customerID', inplace=True)
    df['PaymentMethod'] = df['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
    return df


# Fetch data.
df_telco = pd.read_csv(DATASET_URL)

# Initial preprocessing.
df_telco = preprocess(df_telco)

# Train-test split.
X_train, X_test, Y_train, Y_test = train_test_split(df_telco.drop(columns=TARGET_COLUMN_NAME),
                                                    df_telco.loc[:, TARGET_COLUMN_NAME], 
                                                    random_state=RANDOM_SEED)
