#date: 2025-12-23T16:50:42Z
#url: https://api.github.com/gists/2aa6f6ed87e5b30ce91e1b4983ca25b5
#owner: https://api.github.com/users/hugoM69

def plot_forecast(df, location, mode):
    fig, ax = plt.subplots(figsize=(12, 6))

    for src in df["source"].unique():
        d = df[df["source"] == src]
        ax.plot(d["time"], d["temp"], label=src)

    env = df.groupby("time")["temp"]
    ax.fill_between(
        env.min().index,
        env.min().values,
        env.max().values,
        alpha=0.2,
        label="Optimista–Pesimista"
    )

    ax.set_title(f"Previsión {mode} – {location.capitalize()}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura [°C]")
    ax.legend()
    ax.grid()

    return fig
