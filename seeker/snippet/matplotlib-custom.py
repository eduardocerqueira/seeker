#date: 2024-07-16T17:09:18Z
#url: https://api.github.com/gists/2dc9d21036134946af5efb0f737cdeb1
#owner: https://api.github.com/users/docsallover

import matplotlib.pyplot as plt

# Sample data (same as previous example)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
temperatures = [10, 15, 20, 25, 30, 28]

# Create the line chart with customization
plt.plot(months, temperatures, color='skyblue', marker='o', linestyle='dashed')

# Add labels and title (same as previous example)
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Average Monthly Temperatures")

# Add gridlines for better readability
plt.grid(True)

# Display the plot
plt.show()