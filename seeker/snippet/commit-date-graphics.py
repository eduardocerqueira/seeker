#date: 2025-10-07T17:04:26Z
#url: https://api.github.com/gists/7ad6f0e477c7a5dd3987ea1ac2153557
#owner: https://api.github.com/users/Kambaa

import pandas as pd
import sys
import matplotlib.pyplot as plt

# git log --pretty=format:"%ad" --date=iso > ~/Downloads/commit.txt
# alias gitlog='git log --pretty=format:"%ad" --date=iso > ~/Downloads/commit.txt'
# === Step 1: Full commit times list ===
# commit_times = [line.strip() for line in sys.stdin if line.strip()]

with open("commit.txt", "r", encoding="utf-8") as f:
    commit_times = [line.strip() for line in f if line.strip()]

# === Step 2: Convert to DataFrame ===
df = pd.DataFrame({
    "raw": commit_times,
    "commit_time_utc": pd.to_datetime(commit_times, utc=True, errors="coerce")
})
df = df.dropna(subset=["commit_time_utc"]).copy()

# convert to Istanbul time (UTC+3, handles offsets in log correctly)
df["local_time"] = df["commit_time_utc"].dt.tz_convert("Europe/Istanbul")

# extract useful parts
df["date"] = df["local_time"].dt.date
df["hour"] = df["local_time"].dt.hour
df["weekday"] = df["local_time"].dt.weekday   # Mon=0 ... Sun=6
df["off_hours"] = (
    (df["hour"] < 9) |
    (df["hour"] >= 18) |
    (df["weekday"] >= 5)
)

# === Step 3: Plot calendar-style scatter ===
plt.figure(figsize=(14,8))

# Work hours (Mon–Fri, 09–18)
plt.scatter(df["hour"][~df["off_hours"]], df["date"][~df["off_hours"]],
            c="skyblue", label="Work Hours (Mon–Fri, 09–18)", alpha=0.7)

# Off hours and weekends
plt.scatter(df["hour"][df["off_hours"]], df["date"][df["off_hours"]],
            c="orange", label="Off Hours / Weekend", alpha=0.7)

# Shade weekends with different colors
unique_dates = sorted(df["date"].unique())
for d in unique_dates:
    wd = pd.to_datetime(d).weekday()
    if wd == 5:  # Saturday
        plt.axhspan(d - pd.Timedelta(hours=12),
                    d + pd.Timedelta(hours=12),
                    color="lightpink", alpha=0.3, label="Saturday" if "Saturday" not in plt.gca().get_legend_handles_labels()[1] else "")
    elif wd == 6:  # Sunday
        plt.axhspan(d - pd.Timedelta(hours=12),
                    d + pd.Timedelta(hours=12),
                    color="lightyellow", alpha=0.3, label="Sunday" if "Sunday" not in plt.gca().get_legend_handles_labels()[1] else "")

# Set X-axis ticks (hours)
plt.xticks(range(0,24,2))
plt.xlabel("Hour of Day")

# Set Y-axis ticks to every unique date
plt.yticks(unique_dates)  # Display every unique date
plt.ylabel("Date")

plt.title("Commit Calendar: Work vs Off-Hours & Weekends", fontsize=16)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# === Step 4: Save chart ===
plt.savefig("commit_calendar.png", dpi=300)
print("Commit calendar saved as commit_calendar.png")

# Show interactively (optional)
plt.show()
