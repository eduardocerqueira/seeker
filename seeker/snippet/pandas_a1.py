#date: 2024-05-14T16:59:21Z
#url: https://api.github.com/gists/b74c268d31c0c9155970ee1b03ce54b5
#owner: https://api.github.com/users/ricber

"""
Write a Python script to analyze the data within the provided (dummy) CSV file.
Using Pandas, you should perform the following operations:
1. Load the CSV file and display the first few rows to confirm data reading.
2. Find the sum of all the sales, and total number of transactions in the dataset.
3. Filter the data to extract specific information, such as the total sales of the ProductA.
4. Aggregate the data by product category and calculate descriptive statistics (mean, median, standard deviation) of sales for each category.
5. Perform a time trend analysis, e.g., evaluate monthly sales. You should use the Date column in the proper format and built-in function to manipulate the time-series data.
Data Example
Date,		Product,	Category,	Sales
2023-01-01,	ProductA,	Category1,	100
2023-01-02,	ProductB,	Category2,	150
...
"""

import pandas as pd

# Load the CSV file
file_path = 'sales_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the file
print("First few rows of the file:")
print(data.head())

# Data analysis
total_sales = data['Sales'].sum()

total_transactions = data.shape[0]
print("Analysis Results:")
print(f"\tTotal sales: {total_sales}")
print(f"\tTotal transactions: {total_transactions}")


# Data filtering
product = "ProductA"
product_sales = data[data['Product'] == product]['Sales'].sum()

print("\nSpecific Information:")
print(f"Total sales for 'ProductA': {product_sales}")


# Sales analysis by product category
sales_by_category = data.groupby('Category')
print("\nSales by category:")
print(sales_by_category['Sales'].describe())

#  Analysis of Sales temporal trend by month  
print("\nMonthly trend of sales:")
# Transform the Date column into the proper format
data['Date'] = pd.to_datetime(data['Date'])
# Get the dates into monthly format
month_groups = data['Date'].dt.to_period('M')
# Group using the month groups and sum to get the monthly sales
data.groupby(month_groups)['Sales'].sum()