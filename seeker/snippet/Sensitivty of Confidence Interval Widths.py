#date: 2024-05-13T16:46:43Z
#url: https://api.github.com/gists/da2dea9fef8a5291f943704b9b0583d7
#owner: https://api.github.com/users/srkim

import numpy as np
import matplotlib.pyplot as plt

# Assuming conf_int_95_avg and conf_int_99_avg are previously computed
# Extract the widths of the confidence intervals for proportions (not scaled by sample size)
ci_95_widths_proportional = conf_int_95_avg[:, 1] - conf_int_95_avg[:, 0]
ci_99_widths_proportional = conf_int_99_avg[:, 1] - conf_int_99_avg[:, 0]

# Adjust sample sizes and corresponding CI widths to start from 250
sample_sizes_adjusted = sample_sizes[sample_sizes >= 250]
ci_95_widths_proportional_adjusted = ci_95_widths_proportional[sample_sizes >= 250]
ci_99_widths_proportional_adjusted = ci_99_widths_proportional[sample_sizes >= 250]

# Plotting the proportional widths starting from sample size 250
plt.figure(figsize=(12, 8))
plt.plot(sample_sizes_adjusted, ci_95_widths_proportional_adjusted, label='Width of 95% CI', marker='o')
plt.plot(sample_sizes_adjusted, ci_99_widths_proportional_adjusted, label='Width of 99% CI', marker='o', linestyle='--')
plt.xlabel('Sample Size')
plt.ylabel('Width of Confidence Interval')
plt.title('Sensitivity of Confidence Interval Widths to Sample Size')
plt.legend()
plt.grid(True)
plt.show()