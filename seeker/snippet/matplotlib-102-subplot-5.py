#date: 2022-11-11T17:04:10Z
#url: https://api.github.com/gists/ea18433b701003538564e46d13f3888a
#owner: https://api.github.com/users/audhiaprilliant

# ---------- SUBPLOT WITH GRIDSPEC ----------

# Figure size
fig = plt.figure(figsize = (10, 8))

# Create a grid for plots (2 rows & 3 columns)
gs = gridspec.GridSpec(nrows = 2, ncols = 3, figure = fig)

ax1 = fig.add_subplot(gs[0, :]);
ax1.set_title('# Customer by payment method');

ax2 = fig.add_subplot(gs[1, 0]);
ax2.set_title('# Customer by gender');

ax3 = fig.add_subplot(gs[1, 1]);
ax3.set_title('# Customer by paperless bill status');

ax4 = fig.add_subplot(gs[1, 2]);
ax4.set_title('# Customer by churn status');

# Bar plot - 1
ax1.bar(
    x = 'PaymentMethod',
    height = 'customerID',
    data = df_group_1,
    color = 'orange'
);

# Bar plot - 2
ax2.bar(
    x = 'gender',
    height = 'customerID',
    data = df_group_2,
    color = 'red'
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


# Remove the overlaps between the axis names and the titles
plt.tight_layout();