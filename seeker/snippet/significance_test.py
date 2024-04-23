#date: 2024-04-23T17:02:45Z
#url: https://api.github.com/gists/5a140353dd71cdedb0d5305468f9251c
#owner: https://api.github.com/users/arpitmailgun

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Separate data for rule_based and ml_based model types
rule_data = combined_data[combined_data['model_type'] == 'rule_based'].sample(n=sample_size_rule, random_state=1)
ml_data = combined_data[combined_data['model_type'] == 'ml_based'].sample(n=sample_size_ml, random_state=1)

# Calculate number of conversions and total leads for each model type
rule_converted = rule_data['converted'].sum()
ml_converted = ml_data['converted'].sum()
rule_total = len(rule_data)
ml_total = len(ml_data)

# Perform two-sample proportion z-test
count = np.array([rule_converted, ml_converted])
nobs = np.array([rule_total, ml_total])
z_stat, p_value = proportions_ztest(count, nobs)

print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")

# Interpret the results based on the p-value
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis - There is a significant difference in conversion rates between model types.")
else:
    print("Fail to reject the null hypothesis - There is no significant difference in conversion rates between model types.")