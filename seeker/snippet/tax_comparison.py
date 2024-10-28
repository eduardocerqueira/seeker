#date: 2024-10-28T16:53:19Z
#url: https://api.github.com/gists/ecf88b02a25f7d4bf7e281f040e26d73
#owner: https://api.github.com/users/gezakerecsenyi

import numpy as np
import matplotlib.pyplot as plt

# Define income range and intervals
income_gbp = np.arange(10000, 205000, 5000)
income_huf = income_gbp * 486  # Convert GBP to HUF based on 1 GBP = 486 HUF

# Tax rate constants
# Income tax rates
uk_income_tax_bands = [(50270, 0.2), (125140, 0.4), (np.inf, 0.45)]
hu_income_tax_rate = 0.15

# Social contributions
hu_social_contrib = 0.185  # Hungary social security contribution
uk_social_contrib_lower = 0.12  # UK NI contribution on income over £12,570
uk_social_contrib_higher = 0.02  # UK NI contribution on income above £50,270

# Council tax in the UK as a function of income level (estimated averages)
uk_council_tax_estimate = np.piecewise(income_gbp, 
                                       [income_gbp < 50000, 
                                        (income_gbp >= 50000) & (income_gbp < 100000),
                                        income_gbp >= 100000],
                                       [1200, 2000, 3500])

# Estimate property values as a factor of income
property_value_estimate_uk = income_gbp * 5
property_value_estimate_hu = income_huf * 4

# SDLT (UK) as a function of property value
def calculate_sdlt(property_value):
    if property_value <= 125000:
        return 0
    elif property_value <= 250000:
        return (property_value - 125000) * 0.02
    elif property_value <= 925000:
        return 2500 + (property_value - 250000) * 0.05
    elif property_value <= 1500000:
        return 36250 + (property_value - 925000) * 0.1
    else:
        return 93850 + (property_value - 1500000) * 0.12

# Tax calculations
def calculate_uk_income_tax(income):
    tax = 0
    previous_band_limit = 0
    for band_limit, rate in uk_income_tax_bands:
        if income > previous_band_limit:
            taxable_amount = min(income, band_limit) - previous_band_limit
            tax += taxable_amount * rate
            previous_band_limit = band_limit
        else:
            break
    return tax

# Compute total tax burden as a proportion of income
tax_burden_uk_resident = []
tax_burden_hu_resident = []

for income in income_gbp:
    # UK resident - UK national
    uk_income_tax = calculate_uk_income_tax(income)
    uk_ni = max(0, (income - 12570) * uk_social_contrib_lower) + max(0, (income - 50270) * uk_social_contrib_higher)
    sdlt = calculate_sdlt(income * 5) / (income * 5) if income * 5 > 125000 else 0  # Annualized SDLT based on property value
    total_tax_uk_resident = uk_income_tax + uk_ni + uk_council_tax_estimate[np.where(income_gbp == income)][0] + sdlt * income
    tax_burden_uk_resident.append(total_tax_uk_resident / income)
    
    # Hungarian resident - Hungarian national
    hu_income_tax = income * hu_income_tax_rate
    hu_social_sec = income * hu_social_contrib
    total_tax_hu_resident = hu_income_tax + hu_social_sec
    tax_burden_hu_resident.append(total_tax_hu_resident / income)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(income_gbp, tax_burden_uk_resident, label='UK Resident - UK National', color='blue')
plt.plot(income_gbp, tax_burden_hu_resident, label='Hungarian Resident - Hungarian National', color='green')
plt.xlabel('Gross Annual Income (£)')
plt.ylabel('Proportion of Income Paid in Tax')
plt.title('Proportion of Income Paid in Tax by Income Level')
plt.legend()
plt.grid(True)
plt.show()