#date: 2024-07-11T17:11:02Z
#url: https://api.github.com/gists/cbe211b67ecf6198e06b188df64867d7
#owner: https://api.github.com/users/farzonl

import matplotlib.pyplot as plt

# Data for unemployment rates by month for each year
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

unemployment_2020 = [3.6, 3.5, 4.4, 14.8, 13.2, 11.0, 10.2, 8.4, 7.8, 6.8, 6.7, 6.7]
unemployment_2021 = [6.4, 6.2, 6.1, 6.1, 5.8, 5.9, 5.4, 5.1, 4.7, 4.5, 4.1, 3.9]
unemployment_2022 = [4.0, 3.8, 3.6, 3.7, 3.6, 3.6, 3.5, 3.6, 3.5, 3.6, 3.6, 3.5]
unemployment_2023 = [3.4, 3.6, 3.5, 3.4, 3.7, 3.6, 3.5, 3.8, 3.8, 3.8, 3.7, 3.7]
unemployment_2024 = [3.7, 3.9, 3.8, 3.9, 4.0, 4.1]

# Plotting the data
plt.figure(figsize=(12, 6))

plt.plot(months, unemployment_2020, marker='o', label='2020')
plt.plot(months, unemployment_2021, marker='o', label='2021')
plt.plot(months, unemployment_2022, marker='o', label='2022')
plt.plot(months, unemployment_2023, marker='o', label='2023')
plt.plot(months[:len(unemployment_2024)], unemployment_2024, marker='o', label='2024')

plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
plt.title('Monthly Unemployment Rate (2020-2024)')
plt.legend()
plt.grid(True)
plt.show()
