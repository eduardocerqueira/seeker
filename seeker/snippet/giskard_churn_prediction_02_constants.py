#date: 2023-06-12T16:51:26Z
#url: https://api.github.com/gists/8403b90dfb3b8503850fa62135bcb72a
#owner: https://api.github.com/users/AbSsEnT

DATASET_URL = "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET_COLUMN_NAME = "Churn"
COLUMN_TYPES = {'gender': "category",
                'SeniorCitizen': "category",
                'Partner': "category",
                'Dependents': "category",
                'tenure': "numeric",
                'PhoneService': "category",
                'MultipleLines': "category",
                'InternetService': "category",
                'OnlineSecurity': "category",
                'OnlineBackup': "category",
                'DeviceProtection': "category",
                'TechSupport': "category",
                'StreamingTV': "category",
                'StreamingMovies': "category",
                'Contract': "category",
                'PaperlessBilling': "category",
                'PaymentMethod': "category",
                'MonthlyCharges': "numeric",
                'TotalCharges': "numeric",
                TARGET_COLUMN_NAME: "category"}
FEATURE_TYPES = {i:COLUMN_TYPES[i] for i in COLUMN_TYPES if i != TARGET_COLUMN_NAME}

RANDOM_SEED = 123