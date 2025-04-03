#date: 2025-04-03T16:50:18Z
#url: https://api.github.com/gists/64b76f6ca787c8abe66fbde8fb9df069
#owner: https://api.github.com/users/carloocchiena

import matplotlib.dates as mdates

#  Create a heatmap
heatmap_data = df.copy()
heatmap_data['Date'] = heatmap_data.index.date
heatmap_data['Hour'] = heatmap_data.index.hour

pivot = heatmap_data.pivot_table(index='Date', columns='Hour', values='Prezzo')

# 2. Plot della heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': '€/MWh'})
plt.title('Heatmap Prezzo Unico Nazionale - Orario su base Giornaliera (2024)', fontsize=14)
plt.xlabel('Hour of Day')
plt.ylabel('Date')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()