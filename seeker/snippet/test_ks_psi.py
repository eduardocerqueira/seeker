#date: 2024-06-14T16:47:27Z
#url: https://api.github.com/gists/6760d20405644744d12a8047e277fb69
#owner: https://api.github.com/users/IbLahlou

import numpy as np

# KS Test
def ks_test(train_data, test_data):
    # Combine the data and sort it
    data_all = np.concatenate([train_data, test_data])
    data_all.sort()
    
    # Calculate the empirical CDFs
    cdf_train = np.searchsorted(train_data, data_all, side='right') / len(train_data)
    cdf_test = np.searchsorted(test_data, data_all, side='right') / len(test_data)
    
    # Calculate the KS statistic
    ks_statistic = np.max(np.abs(cdf_train - cdf_test))
    
    # Calculate the p-value (approximation for large samples)
    n1 = len(train_data)
    n2 = len(test_data)
    en = np.sqrt(n1 * n2 / (n1 + n2))
    p_value = 2 * np.exp(-2 * (ks_statistic * en) ** 2)
    
    return ks_statistic, p_value

# PSI Calculation
def calculate_psi(base, current, bins=10):
    # Calculate the percent distributions
    base_counts, bin_edges = np.histogram(base, bins=bins)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    base_percents = base_counts / len(base)
    current_percents = current_counts / len(current)

    # Replace 0 counts to avoid division by zero or log of zero errors
    base_percents = np.where(base_percents == 0, 0.0001, base_percents)
    current_percents = np.where(current_percents == 0, 0.0001, current_percents)

    # Calculate PSI
    psi = np.sum((current_percents - base_percents) * np.log(current_percents / base_percents))
    return psi

# Example usage
train_data = np.random.normal(0, 1, 1000)
test_data = np.random.normal(0.5, 1, 1000)

ks_stat, ks_p_value = ks_test(train_data, test_data)
psi_value = calculate_psi(train_data, test_data)

print(f'KS Statistic: {ks_stat}, P-value: {ks_p_value}')
print(f'PSI: {psi_value}')
