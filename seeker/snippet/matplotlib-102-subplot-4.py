#date: 2022-11-11T17:03:08Z
#url: https://api.github.com/gists/51934e749716ba8ebb0f5b04928e6b36
#owner: https://api.github.com/users/audhiaprilliant

# ---------- SUBPLOT (2 ROWS, 2 COLUMNS) ----------

# Create a grid for plots (2 rows & 2 columns)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 6))

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

# Bar plot - 3
ax3.bar(
    x = 'PaperlessBilling',
    height = 'customerID',
    data = df_group_4,
    color = 'green'
);

# Bar plot - 4
ax4.bar(
    x = 'Churn',
    height = 'customerID',
    data = df_group_5,
    color = 'purple'
);

# Titles
ax1.set_title('# Customer by gender');
ax2.set_title('# Customer by contract');
ax3.set_title('# Customer by paperless billing status');
ax4.set_title('# Customer by churn status');

# Title the figure
plt.suptitle('# Customer in Telco XYZ', fontsize = 14, fontweight = 'bold');

# Remove the overlaps between the axis names and the titles
plt.tight_layout();