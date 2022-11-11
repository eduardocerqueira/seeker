#date: 2022-11-11T16:59:19Z
#url: https://api.github.com/gists/98923124353474c752e64145b949192a
#owner: https://api.github.com/users/audhiaprilliant

# ---------- SUBPLOT (1 ROWS, 2 COLUMNS) ----------

# Create a grid for plots (1 row & 2 columns)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4.8))

# Bar plot - 1
ax[0].bar(
    x = 'gender',
    height = 'customerID',
    data = df_group_2,
    color = 'red'
)
# Bar plot - 2
ax[1].bar(
    x = 'Contract',
    height = 'customerID',
    data = df_group_3,
    color = 'blue'
);

# Titles
ax[0].set_title('# Customer by gender');
ax[1].set_title('# Customer by contract');

# Title the figure
plt.suptitle('# Customer in Telco XYZ', fontsize = 14, fontweight = 'bold');