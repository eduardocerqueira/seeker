#date: 2025-05-23T17:10:29Z
#url: https://api.github.com/gists/f748454f5b0b0e272287200ad2d36dd7
#owner: https://api.github.com/users/anix-lynch

import pandas as pd
import pyarrow  # required for Parquet I/O

# 1. Create a small DataFrame (like your CSV sample)
df = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "department": ["Engineering", "Marketing"],
    "salary": [80000, 72000]
})

# 2. Save as CSV (row-based)
df.to_csv("employees.csv", index=False)

# 3. Save as Parquet (column-based)
df.to_parquet("employees.parquet", index=False)

# 4. Load just the salary column from Parquet
salary_only = pd.read_parquet("employees.parquet", columns=["salary"])

print("💼 Full DataFrame from CSV:")
print(df)
print("\n🧃 Only 'salary' column from Parquet:")
print(salary_only)
