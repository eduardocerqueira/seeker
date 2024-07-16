#date: 2024-07-16T16:53:46Z
#url: https://api.github.com/gists/83d8b29ab8ad3042608c1bfebfb3600f
#owner: https://api.github.com/users/docsallover

import matplotlib.pyplot as plt

# Sample data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
temperatures = [10, 15, 20, 25, 30, 28]

# Create the line chart
plt.plot(months, temperatures)

# Add labels and title
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Average Monthly Temperatures")

# Display the plot
plt.show()