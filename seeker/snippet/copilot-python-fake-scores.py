#date: 2024-07-15T16:55:28Z
#url: https://api.github.com/gists/48fff68259a7c7dd0e406cbad5d86e37
#owner: https://api.github.com/users/summerofgeorge

# Use Python and the faker package to create a synthetic Excel dataset for an organization's employee performance review for 500 employees with the following details:

# - Columns: Employee ID, Department, Performance Score, Salary, Years of Experience
# - Performance Score: Normally distributed with a mean of 70 and a standard deviation of 10
# - Salary: Log-normally distributed with a mean of $50,000 and a standard deviation of $15,000
# - Years of Experience: Exponentially distributed with a lambda of 0.1
# - Department: Randomly chosen from 'Sales', 'HR', 'IT', 'Marketing', 'Finance
# - Random seed: Set to 1234.

import pandas as pd
import numpy as np
from faker import Faker

# Set the random seed
np.random.seed(1234)

# Initialize the faker generator
fake = Faker()

# Generate employee data
employee_data = {
    'Employee ID': [fake.unique.random_number(digits=6) for _ in range(500)],
    'Department': [fake.random_element(['Sales', 'HR', 'IT', 'Marketing', 'Finance']) for _ in range(500)],
    'Performance Score': np.random.normal(loc=70, scale=10, size=500),
    'Salary': np.random.lognormal(mean=np.log(50000), sigma=np.log(15000), size=500),
    'Years of Experience': np.random.exponential(scale=1/0.1, size=500)
}

# Create a DataFrame
df = pd.DataFrame(employee_data)

# Save to an Excel file
df.to_excel('employee_performance.xlsx', index=False)
print("Excel workbook 'employee_performance.xlsx' created successfully!")
