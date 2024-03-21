#date: 2024-03-21T17:07:46Z
#url: https://api.github.com/gists/2593cab3927fdbacef1e88d0409e6cd3
#owner: https://api.github.com/users/arpitmailgun

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.power import TTestIndPower

# Define conversion rates and desired effect size
baseline_conversion_rate = 0.1
desired_effect_size = 0.1
alpha = 0.05
power = 0.8

# Calculate required sample size per group
number_of_leads_per_group = TTestIndPower().solve_power(
    effect_size=desired_effect_size,
    alpha=alpha,
    power=power,
    nobs1=None,
    ratio=1
)