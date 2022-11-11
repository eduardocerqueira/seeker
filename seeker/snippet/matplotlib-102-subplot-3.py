#date: 2022-11-11T17:02:11Z
#url: https://api.github.com/gists/9b9eecb34649a944ef459561d3b3d974
#owner: https://api.github.com/users/audhiaprilliant

# ---------- SUBPLOT (2 ROWS, 2 COLUMNS) ----------

# Create a grid for plots (2 rows & 2 columns)
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 6))

# Bar plot - 1
ax[0, 0].bar(
    x = 'gender',
    height = 'customerID',
    data = df_group_2,
    color = 'red'
)

# Bar plot - 2
ax[0, 1].bar(
    x = 'Contract',
    height = 'customerID',
    data = df_group_3,
    color = 'blue'
);
# Set x-ticks
ax[0, 1].set_xticks([0, 1]);

# Bar plot - 3
ax[1, 0].bar(
    x = 'PaperlessBilling',
    height = 'customerID',
    data = df_group_4,
    color = 'green'
);

# Bar plot - 4
ax[1, 1].bar(
    x = 'Churn',
    height = 'customerID',
    data = df_group_5,
    color = 'purple'
);

# Titles
ax[0, 0].set_title('# Customer by gender');
ax[0, 1].set_title('# Customer by contract');
ax[1, 0].set_title('# Customer by paperless billing status');
ax[1, 1].set_title('# Customer by churn status');

# Title the figure
plt.suptitle('# Customer in Telco XYZ', fontsize = 14, fontweight = 'bold');

# Remove the overlaps between the axis names and the titles
plt.tight_layout();