#date: 2022-11-11T17:00:36Z
#url: https://api.github.com/gists/27c7dd01fd919dcaec042898f288bd83
#owner: https://api.github.com/users/audhiaprilliant

# ---------- SUBPLOT (1 ROWS, 2 COLUMNS) ----------

# Create a grid for plots (1 rows & 2 columns)
fig, ((ax1, ax2)) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4.8))

# Bar plot - 1
ax1.bar(
    x = 'gender',
    height = 'customerID',
    data = df_group_2,
    color = 'red'
)

# Bar plot - 2
ax2.bar(
    x = 'Contract',
    height = 'customerID',
    data = df_group_3,
    color = 'blue'
);

# Titles
ax1.set_title('# Customer by gender');
ax2.set_title('# Customer by contract');

# Title the figure
plt.suptitle('# Customer in Telco XYZ', fontsize = 14, fontweight = 'bold');