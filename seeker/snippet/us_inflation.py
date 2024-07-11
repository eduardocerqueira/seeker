#date: 2024-07-11T17:01:03Z
#url: https://api.github.com/gists/d6c9c84f55fbc21fa8edcba7d56e0a88
#owner: https://api.github.com/users/farzonl

import matplotlib.pyplot as plt

# Data for inflation rates by month for each year
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

inflation_2024 = [3.1, 3.2, 3.5, 3.4, 3.3, 3.0]
inflation_2023 = [6.4, 6.0, 5.0, 4.9, 4.0, 3.0, 3.2, 3.7, 3.7, 3.2, 3.1, 3.4]
inflation_2022 = [7.5, 7.9, 8.5, 8.3, 8.6, 9.1, 8.5, 8.3, 8.2, 7.7, 7.1, 6.5]
inflation_2021 = [1.4, 1.7, 2.6, 4.2, 5.0, 5.4, 5.4, 5.3, 5.4, 6.2, 6.8, 7.0]

# Plotting the data
plt.figure(figsize=(12, 6))

plt.plot(months[:len(inflation_2024)], inflation_2024, marker='o', label='2024')
plt.plot(months, inflation_2023, marker='o', label='2023')
plt.plot(months, inflation_2022, marker='o', label='2022')
plt.plot(months, inflation_2021, marker='o', label='2021')

plt.xlabel('Month')
plt.ylabel('Inflation Rate (%)')
plt.title('Monthly Inflation Rate (2021-2024)')
plt.legend()
plt.grid(True)
plt.show()
