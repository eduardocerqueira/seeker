#date: 2022-11-11T16:56:46Z
#url: https://api.github.com/gists/bed1d8449389beb44ccabd3df896cdeb
#owner: https://api.github.com/users/audhiaprilliant

# ---------- MULTIPLOT ----------

# Figure size
fig = plt.figure(figsize = (10, 4.8))

# Bar plot
plt.bar(
    x = 'PaymentMethod',
    height = 'customerID',
    data = df_group_1,
    color = 'red'
)
# Line plot
plt.plot(
    'PaymentMethod',
    'CummulativeSum',
    data = df_group_1,
    color = 'blue',
    marker = 'o'
);