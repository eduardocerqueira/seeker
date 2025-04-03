#date: 2025-04-03T16:47:46Z
#url: https://api.github.com/gists/703f5946ca583151762048a91e671040
#owner: https://api.github.com/users/carloocchiena

# Boxplot by hour of day

plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='Hour', y='Prezzo')
plt.title('Hourly Distribution of PUN Prices')
plt.xlabel('Hour of Day')
plt.ylabel('€/MWh')
plt.grid(True)
plt.tight_layout()
plt.show()