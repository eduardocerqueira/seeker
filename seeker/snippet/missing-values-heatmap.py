#date: 2024-05-22T17:01:10Z
#url: https://api.github.com/gists/bcdbf9904538884ce900f5db49b56d23
#owner: https://api.github.com/users/summerofgeorge

sns.heatmap(penguins_df.isna(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values in Penguins Dataset')