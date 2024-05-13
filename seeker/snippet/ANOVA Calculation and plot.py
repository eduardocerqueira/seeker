#date: 2024-05-13T16:54:36Z
#url: https://api.github.com/gists/d625c3f0f7c1f4a412295f48748e17b4
#owner: https://api.github.com/users/srkim

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data from an Excel file
file_path = 'VSVC1A.xlsx'
data = pd.read_excel(file_path)

# Prepare data for ANOVA by melting the dataframe
anova_data = pd.melt(data, id_vars='Date', value_vars=['Burgiss Mean', 'VSVC I', 'Burgiss Median'], var_name='Group', value_name='Value')

# Perform ANOVA
anova_model = ols('Value ~ C(Group)', data=anova_data).fit()
anova_results = sm.stats.anova_lm(anova_model, typ=1)

# Print F-Statistic and P-Value
print("F-Statistic:", round(anova_results.loc['C(Group)', 'F'], 2))
print("P-Value:", round(anova_results.loc['C(Group)', 'PR(>F)'], 2))

# Manually calculate KDE for each group
groups = anova_data['Group'].unique()
colors = ['blue', 'orange', 'green']  # Colors for each group
plt.figure(figsize=(10, 6))

for i, group in enumerate(groups):
    subset = anova_data[anova_data['Group'] == group]['Value']
    density = sm.nonparametric.KDEUnivariate(subset)
    density.fit()
    plt.plot(density.support, density.density, fillstyle='bottom', alpha=0.5, color=colors[i], label=group)

plt.title('Manual KDE Plot for Burgess Mean, VSVC I, and Burgess Median')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()