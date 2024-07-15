#date: 2024-07-15T16:58:40Z
#url: https://api.github.com/gists/56abe14f64be817384c90b579f147268
#owner: https://api.github.com/users/summerofgeorge

import pandas as pd
import numpy as np
from faker import Faker

# Set the random seed
np.random.seed(1234)

# Initialize the faker generator
fake = Faker()

# Generate inventory data
inventory_data = {
    'Item ID': [fake.unique.random_number(digits=6) for _ in range(2000)],
    'Category': [fake.random_element(['Electronics', 'Clothing', 'Home Goods', 'Sports Equipment', 'Toys']) for _ in range(2000)],
    'Stock Level': np.random.normal(loc=100, scale=30, size=2000),
    'Reorder Level': np.random.uniform(low=20, high=50, size=2000),
    'Lead Time': np.random.exponential(scale=1/0.05, size=2000)
}

# Create a DataFrame
df = pd.DataFrame(inventory_data)

# Save to an Excel file
df.to_excel('warehouse_inventory.xlsx', index=False)
print("Excel workbook 'warehouse_inventory.xlsx' created successfully!")
