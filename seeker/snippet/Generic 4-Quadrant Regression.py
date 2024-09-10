#date: 2024-09-10T16:44:44Z
#url: https://api.github.com/gists/01e1f2f2fd5a413db7dbd68df088489a
#owner: https://api.github.com/users/srkim

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Read data from an Excel file and store it in a Pandas DataFrame
df = pd.read_excel('CorrelationsHedge2.xlsx')

# Define the response and explanatory variables
y = df['Y']
x = df['X']

# Calculate medians and IQRs for x and y
median_x = np.median(x)
median_y = np.median(y)
q1_x, q3_x = np.percentile(x, [25, 75])
q1_y, q3_y = np.percentile(y, [25, 75])
iqr_x = q3_x - q1_x
iqr_y = q3_y - q1_y

# Identify outliers
outliers = (x < q1_x - 1.5 * iqr_x) | (x > q3_x + 1.5 * iqr_x) | (y < q1_y - 1.5 * iqr_y) | (y > q3_y + 1.5 * iqr_y)

# Calculate the slope, intercept, and R-squared value of the regression line
slope, intercept, r_value, _, _ = linregress(x, y)
r_squared = r_value**2

# Generate the regression line
regression_line = slope * x + intercept

# Divide data into quadrants counterclockwise from the top right
q1 = (x >= 0) & (y >= 0)  # Top right
q2 = (x <= 0) & (y >= 0)  # Top left
q3 = (x <= 0) & (y <= 0)  # Bottom left
q4 = (x >= 0) & (y <= 0)  # Bottom right

# Calculate correlations for each quadrant
corr_q1 = np.corrcoef(x[q1], y[q1])[0, 1]
corr_q2 = np.corrcoef(x[q2], y[q2])[0, 1]
corr_q3 = np.corrcoef(x[q3], y[q3])[0, 1]
corr_q4 = np.corrcoef(x[q4], y[q4])[0, 1]

# Calculate overall correlation
overall_corr = np.corrcoef(x, y)[0, 1]

# Calculate the number of points in each quadrant and the percentage of total points
total_points = len(x)
q1_count = np.sum(q1)
q2_count = np.sum(q2)
q3_count = np.sum(q3)
q4_count = np.sum(q4)

q1_percentage = (q1_count / total_points) * 100
q2_percentage = (q2_count / total_points) * 100
q3_percentage = (q3_count / total_points) * 100
q4_percentage = (q4_count / total_points) * 100

# Calculate the number of points where x is greater than y in each quadrant
q1_x_greater_y = np.sum((q1) & (x > y))
q2_x_greater_y = np.sum((q2) & (x > y))
q3_x_greater_y = np.sum((q3) & (x > y))
q4_x_greater_y = np.sum((q4) & (x > y))

# Calculate the percentage of points where x is greater than y in each quadrant
q1_x_greater_y_percentage = (q1_x_greater_y / q1_count) * 100
q2_x_greater_y_percentage = (q2_x_greater_y / q2_count) * 100
q3_x_greater_y_percentage = (q3_x_greater_y / q3_count) * 100
q4_x_greater_y_percentage = (q4_x_greater_y / q4_count) * 100

# Print summary statistics
print(f"Sum of correlations in quadrants: {corr_q1 + corr_q2 + corr_q3 + corr_q4:.4f}")
print(f"Overall correlation: {overall_corr:.4f}")
print('________________________________________________')

# Print the number of points in each quadrant and the percentage of points from the total points
print(f"Number of points in Q1: {q1_count} ({q1_percentage:.2f}%)")
print(f"Number of points in Q2: {q2_count} ({q2_percentage:.2f}%)")
print(f"Number of points in Q3: {q3_count} ({q3_percentage:.2f}%)")
print(f"Number of points in Q4: {q4_count} ({q4_percentage:.2f}%)")
print('________________________________________________')

# Print the percentage of points where x is greater than y in each quadrant
print(f"Percentage of points where x > y in Q1: {q1_x_greater_y_percentage:.2f}%")
print(f"Percentage of points where x > y in Q2: {q2_x_greater_y_percentage:.2f}%")
print(f"Percentage of points where x > y in Q3: {q3_x_greater_y_percentage:.2f}%")
print(f"Percentage of points where x > y in Q4: {q4_x_greater_y_percentage:.2f}%")

# Create scatter plot with colored quadrants and 'x' marker for outliers
plt.figure(figsize=(12, 8))
plt.scatter(x[q1 & ~outliers], y[q1 & ~outliers], alpha=.6, s=50, label=f'Q1: {corr_q1:.4f}')
plt.scatter(x[q2 & ~outliers], y[q2 & ~outliers], alpha=.6, s=50, label=f'Q2: {corr_q2:.4f}')
plt.scatter(x[q3 & ~outliers], y[q3 & ~outliers], alpha=.6, s=50, label=f'Q3: {corr_q3:.4f}')
plt.scatter(x[q4 & ~outliers], y[q4 & ~outliers], alpha=.6, s=50, label=f'Q4: {corr_q4:.4f}')

# Plot outliers with 'x' marker with the same color as their respective quadrants
plt.scatter(x[q1 & outliers], y[q1 & outliers], s=40, color='C0', marker='x')
plt.scatter(x[q2 & outliers], y[q2 & outliers], s=40, color='C1', marker='x')
plt.scatter(x[q3 & outliers], y[q3 & outliers], s=40, color='C2', marker='x')
plt.scatter(x[q4 & outliers], y[q4 & outliers], s=40, color='C3', marker='x')

# Plot regression line and axis lines
plt.plot(x, regression_line, color="gray", label="Regression Line")
plt.axvline(x=0, color='gray', linestyle='--')
plt.axhline(y=0, color='gray', linestyle='--')

plt.title(f'Y vs. X\nOverall Correlation: {overall_corr:.4f}\nR-squared: {r_squared:.4f}\nSlope: {slope:.4f}: x are Outliers', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Save the plot as an SVG file
plt.savefig('Correlation.svg')

plt.show()