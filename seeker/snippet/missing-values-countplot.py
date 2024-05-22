#date: 2024-05-22T16:55:16Z
#url: https://api.github.com/gists/d0a07396c324632210cb9f713529b673
#owner: https://api.github.com/users/summerofgeorge

penguins_df = xl("penguins[#All]", headers=True)
missing_data = penguins_df.isna().sum() / len(penguins_df)

plt.figure(figsize=(10, 6))
missing_data[missing_data > 0].plot(kind='bar', color='blue', alpha=0.7)
plt.title('Count of Missing Values Per Column in Penguins Dataset')
plt.ylabel('Number of Missing Values')