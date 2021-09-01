#date: 2021-09-01T01:25:53Z
#url: https://api.github.com/gists/392a0aa8868b45d6fe18dda97316d642
#owner: https://api.github.com/users/tommason14

# ^ imports and pre-processing
sns.set(style="white", palette="Set2", font_scale=1.2)
p = sns.relplot(
    x="COM separation (Å)",
    y="energy",
    hue="solv",
    col="system",
    col_wrap=4,
    kind="line",
    ci=None,
    marker="o",
    facet_kws={"sharex": False, "sharey": False},
    data=df,
)
p.set_titles("{col_name}")

# add on vertical lines to show minimum energy
for ax in p.axes.flatten():
    sep = sapt2[sapt2["system"] == ax.title._text]["r_COM"].values[0]
    ax.axvline(sep, ls="--", c="#aaaaaa")