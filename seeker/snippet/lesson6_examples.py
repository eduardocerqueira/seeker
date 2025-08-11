#date: 2025-08-11T17:12:31Z
#url: https://api.github.com/gists/e2bcbe6595b8a095304bb52a9c308383
#owner: https://api.github.com/users/toma63

import pandas as pd

# Sample data
data = {
    'Store': ['North', 'North', 'North', 'South', 'South', 'South'],
    'Product': ['Apple', 'Banana', 'Orange', 'Apple', 'Banana', 'Orange'],
    'Sales': [120, 75, 50, 130, 70, 80],
    'Quantity': [30, 25, 10, 35, 20, 15]
}
df = pd.DataFrame(data)
print(df)
# %%
# See total sales of each product by store.
# pivot on Store (as rows), Product (as columns), and use the sum of Sales
pivot_sales = pd.pivot_table(df, 
                             index='Store', 
                             columns='Product', 
                             values='Sales', 
                             aggfunc='sum')

print(pivot_sales)

# %%
# multiple statistics
pivot_multi = pd.pivot_table(df, 
                             index='Store', 
                             columns='Product', 
                             values='Sales', 
                             aggfunc=['sum', 'mean'])
print(pivot_multi)
# %%
# Multiple indices
# Pivot on Quantity with Store and Product as row indices
pivot_multi_index = pd.pivot_table(df, 
                                   index=['Store', 'Product'], 
                                   values='Quantity', 
                                   aggfunc='sum')
print(pivot_multi_index)
# %%
df = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [4, 5, 6]
})
print(df)

# Returning multiple columns
def compute_ratios(row):
    total = row['A'] + row['B']
    # Return a Series
    return pd.Series({
        'A_ratio': row['A'] / total,
        'B_ratio': row['B'] / total
    })
df[['A_ratio', 'B_ratio']] = df.apply(compute_ratios, axis=1)
print(df)
# %%

df = pd.DataFrame({
    "A": ["10", "20", "30", "40"],
    "B": ["1.5", "2.5", "3.6", "4.1"],
    "C": ["2021-01-01", "2021-02-15", "2021-03-10", "2021-04-20"], 
    "D": ["apple", "banana", "cherry", "banana"],
    "E": ['Beantown', 'Cisco', 'Windy', 'Philly']
})

print("Original DataFrame:")
print(df)

df["A"] = df["A"].astype(int)
df["B"] = pd.to_numeric(df["B"])
df["C"] = pd.to_datetime(df["C"])
df["D"] = df["D"].map(lambda fruit: fruit.upper())
df["E"] = df["E"].replace({'Beantown': 'Boston', 'Cisco': 'San Francisco', 'Windy': 'Chicago', 'Philly': 'Philadelphia'})

print("\nDataFrame after transformations:")
print(df)
# %%
data = {
    'age': [15, 20, 22, 23, 29, 34, 37, 45, 52, 63, 75, 87]
}
df = pd.DataFrame(data)
dfn = df.copy()
print(df)
# 1) Cut into fixed intervals
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                       labels=['Child', 'Young Adult', 'Adult', 'Senior'])
print(df)
# 2) Cut into a specified number of bins (e.g., 3 bins)
dfn['age_qbins'] = pd.qcut(dfn['age'], [0, .15, .4, .75, 1.], 
                           labels=['Child', 'Young Adult', 'Adult', 'Senior'])
print(dfn)
# %%
data = {
    'age': [15, 20, 22, 23, 29, 34, 37, 45, 52, 63, 75, 87]
}
# cut with specified bins
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                       labels=['Child', 'Young Adult', 'Adult', 'Senior'])
# qcut using specified quantiles
dfn['age_qbins'] = pd.qcut(dfn['age'], [0, .15, .4, .75, 1.], 
                           labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# %%
data = {
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Age': [24, 42, 24, 36, 42],
    'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Pittsburgh']
}

new_df = df.drop_duplicates(subset=['Name', 'Age'])

# %%
import pandas as pd
import numpy as np
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah'],
    'Age': [25, 30, 35, 40, 22, 29, 34, 128],
    'Weight': [55, 300, 68, 80, 50, 72, 65, 60],
    'Height': [165, 175, 170, 217, 160, 168, 172, 165]
}

df = pd.DataFrame(data)

print(df)

df['Age'] = df['Age'].apply(lambda x: df['Age'].median() if x > 110 else x)
df['Weight'] = df['Weight'].apply(lambda x: df['Weight'].mean() if x > 120 else x)
df['Height'] = df['Height'].apply(lambda x: df['Height'].median() if x > 210 else x)

print(df)

df['Age'] += 1
df['Age'] = np.add(df['Age'], 1)

print(df)
# %%
age_series = df['Age']
mean = age_series.mean()
std_dev = age_series.std()
cutoff = 2 * std_dev
lower_bound = mean - cutoff
upper_bound = mean + cutoff
print(f'lower bound: {lower_bound:.2f} '
      f'upper bound: {upper_bound:.2f}')
