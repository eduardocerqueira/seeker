#date: 2025-12-17T17:15:18Z
#url: https://api.github.com/gists/3f0c12324a3962165ea2607ee26f641f
#owner: https://api.github.com/users/blaylockbk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

# Replicate Figure 1.9 (page 10) in Wallace and Hobbs 2nd Edition

# Standard atmosphere data (U.S. Standard Atmosphere 1976)
# https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
LEVELS_M = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852, 90000, 120000])
TEMPS_K = np.array(
    [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946, 186.946, 220]
)

# Atmospheric layer definitions
LAYERS = [
    (0, 11, "Troposphere", "#e3f2fd"),
    (11, 50, "Stratosphere", "#bbdefb"),
    (50, 85, "Mesosphere", "#90caf9"),
    (85, 120, "Thermosphere", "#64b5f6"),
]

# Notable altitude features
FEATURES = {
    "Burj Khalifa": 0.83,
    "Mt. Everest": 8.85,
    "Commercial airliner": 11,
    "ER-2 aircraft": 20,
    "Ozone layer peak": 25,
    "Weather balloon": 35,
    "D-layer ionosphere": 75,
    "Meteor ablation": 85,
}

# Scale height for pressure approximation
SCALE_HEIGHT_KM = 7.4


def create_atmosphere_plot():
    """Create enhanced atmospheric profile visualization."""
    # Convert units
    z_km = LEVELS_M / 1000
    T_C = TEMPS_K - 273.15

    # Interpolate for smooth temperature profile
    z_fine = np.linspace(z_km.min(), z_km.max(), 500)
    T_fine = np.interp(z_fine, z_km, T_C)

    # Calculate pressure using scale height approximation
    p_hPa = 1013.25 * np.exp(-z_fine / SCALE_HEIGHT_KM)

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(8, 10), dpi=120)

    # Add colored layer backgrounds
    for z0, z1, name, color in LAYERS:
        ax.add_patch(
            Rectangle(
                (ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -100, z0),
                200,  # Wide enough to cover any temperature range
                z1 - z0,
                facecolor=color,
                alpha=0.3,
                zorder=0,
                edgecolor="none",
            )
        )

    # Plot main temperature profile
    ax.plot(T_fine, z_fine, color="#d32f2f", lw=2.5, label="Temperature", zorder=5)
    ax.scatter(
        T_C, z_km, color="#d32f2f", s=40, zorder=6, edgecolor="white", linewidth=1
    )

    # Layer boundaries
    for zb in [11, 50, 85]:
        ax.axhline(zb, color="#424242", lw=1.2, ls="-", alpha=0.7, zorder=3)

    # Tropopause and stratopause (major boundaries)
    for zb in [20, 32, 47, 51, 71, 84.852]:
        ax.axhline(zb, color="#757575", lw=0.8, ls="--", alpha=0.5, zorder=2)

    # Layer labels
    for z0, z1, name, _ in LAYERS:
        ax.text(
            -95,
            (z0 + z1) / 2,
            name,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#1565c0",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="none"
            ),
        )

    # Feature annotations on the right side
    for label, z in FEATURES.items():
        x_temp = T_fine[np.argmin(np.abs(z_fine - z))]
        x_label = 40  # Fixed position on right side, inside axis

        ax.plot(
            [x_temp, x_label],
            [z, z],
            color="#616161",
            ls=":",
            lw=1,
            alpha=0.6,
            zorder=1,
        )
        ax.scatter(
            x_temp,
            z,
            color="#ff6f00",
            s=30,
            zorder=6,
            marker="o",
            edgecolor="white",
            linewidth=1,
        )
        ax.text(
            x_label + 1,
            z,
            f" {label}",
            va="center",
            ha="right",
            fontsize=8.5,
            color="#212121",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#fff9c4",
                alpha=0.9,
                edgecolor="#fbc02d",
                linewidth=0.5,
            ),
        )

    # Axes configuration
    ax.set_ylim(0, 120)
    ax.set_xlim(-100, 50)
    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude (km)", fontsize=12, fontweight="bold")
    ax.set_yticks(range(0, 121, 10))
    ax.grid(axis="both", alpha=0.3, ls=":", color="#9e9e9e")

    # Secondary temperature axis (Kelvin)
    secax = ax.secondary_xaxis(
        "top", functions=(lambda C: C + 273.15, lambda K: K - 273.15)
    )
    secax.set_xlabel("Temperature (K)", fontsize=12, fontweight="bold")
    secax.tick_params(labelsize=10)

    # Pressure axis
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(p_hPa.max(), p_hPa.min())
    ax2.set_ylabel("Pressure (hPa)", fontsize=12, fontweight="bold")
    ax2.set_yticks([1000, 500, 200, 100, 50, 20, 10, 5, 1, 0.1, 0.01])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.5g}"))
    ax2.tick_params(labelsize=10)

    # Styling
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    for spine in ["left", "right", "bottom"]:
        ax.spines[spine].set_linewidth(1.5)
        ax2.spines[spine].set_linewidth(1.5)

    # Title
    ax.set_title(
        "U.S. Standard Atmosphere (1976)",
        fontsize="xx-large",
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()
    return fig


# Generate the plot
fig = create_atmosphere_plot()
plt.show()
