#date: 2024-07-15T17:02:17Z
#url: https://api.github.com/gists/6218180cf42c9e2297493947befc7031
#owner: https://api.github.com/users/summerofgeorge

import pandas as pd
import numpy as np
from faker import Faker

# Set the random seed
np.random.seed(1234)

# Initialize the faker generator
fake = Faker()

# Generate customer churn data
customer_data = {
    'Customer ID': [fake.unique.random_number(digits=6) for _ in range(5000)],
    'Age': np.random.normal(loc=35, scale=10, size=5000),
    'Tenure': np.random.uniform(low=1, high=72, size=5000),
    'Monthly Charges': np.random.normal(loc=70, scale=20, size=5000),
}

# Calculate Total Charges
customer_data['Total Charges'] = customer_data['Monthly Charges'] * customer_data['Tenure']

# Generate churn (0 or 1) based on probability
customer_data['Churn'] = np.random.choice([0, 1], size=5000, p=[0.8, 0.2])

# Create a DataFrame
df = pd.DataFrame(customer_data)

# Save to an Excel file
df.to_excel('telecom_customer_churn.xlsx', index=False)
print("Excel workbook 'telecom_customer_churn.xlsx' created successfully!")
