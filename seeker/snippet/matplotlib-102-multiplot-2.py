#date: 2022-11-11T16:58:05Z
#url: https://api.github.com/gists/c92a0f631342333a3e53bd51ed1db317
#owner: https://api.github.com/users/audhiaprilliant

# ---------- MULTIPLOT ----------

# Create a grid for plots
fig, ax = plt.subplots(figsize = (10, 5))

# Make another axes object for secondary y-Axis
ax2 = ax.twinx()

# Bar plot
ax.bar(
    x = 'PaymentMethod',
    height = 'customerID',
    data = df_group_1,
    color = 'red'
)
# Line plot
ax2.plot(
    'PaymentMethod',
    'CummulativePerc',
    data = df_group_1,
    color = 'blue',
    marker = 'o'
);

# Plot title
plt.title('Number of Customer by Payment Method', fontsize = 16, fontweight = 'bold', fontstyle = 'italic');
# Vertical axis label (primary)
ax.set_ylabel('# Customer', fontsize = 12, fontstyle = 'italic');
# Vertical axis label (secondary)
ax2.set_ylabel('Percentage (%)', fontsize = 12, fontstyle = 'italic');
# Horizontal axis label
ax.set_xlabel('Payment method', fontsize = 12, fontstyle = 'italic');