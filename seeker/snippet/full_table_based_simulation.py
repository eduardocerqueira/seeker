#date: 2025-12-10T17:14:20Z
#url: https://api.github.com/gists/497f55c09f33b24e6366006b7fadb81e
#owner: https://api.github.com/users/realKfiros

import random
import math
from collections import Counter, defaultdict

# ----- Configuration -----

N_TEAMS = 36          # number of teams in the league phase
ROUNDS = 8            # number of games per team (for reference only)
MAX_POINTS = ROUNDS * 3  # 8 * 3 = 24

N_SIM = 100_000       # number of simulated seasons

# "Corrected" points model:
# We don't simulate each match. Instead, we assume each team's TOTAL points
# are drawn from the same (identical) distribution with more spread.
#
# You can tune these two parameters:
MEAN_POINTS = 11.8    # average total points for a team
STD_POINTS  = 4.5     # standard deviation of total points

# ----- Helper: draw total points for one team -----

def draw_total_points():
    """
    Draws a total points value for a single team from a Normal distribution,
    then clamps it to [0, MAX_POINTS] and rounds to the nearest integer.
    All teams use the same distribution -> no strength differences.
    """
    # Box-Muller to generate N(0, 1)
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    raw_points = MEAN_POINTS + STD_POINTS * z

    # Clamp to valid range
    raw_points = max(0.0, min(float(MAX_POINTS), raw_points))

    # Round to nearest integer
    return int(round(raw_points))

# ----- Statistics storage -----

# results_per_points[points] = {
#   "count":     how many times this point total occurred,
#   "top8":      how many times it resulted in top 8,
#   "top16":     how many times it resulted in top 16,
#   "top24":     how many times it resulted in top 24,
#   "positions": Counter of finishing positions
# }
results_per_points = defaultdict(lambda: {
    "count": 0,
    "top8": 0,
    "top16": 0,
    "top24": 0,
    "positions": Counter(),
})

random.seed(123)  # for reproducibility

# ----- Simulation loop -----

for sim in range(N_SIM):
    # Draw total points for each team in this season
    points = [draw_total_points() for _ in range(N_TEAMS)]

    # Ranking: sort by total points, ties broken randomly
    ranking = sorted(
        range(N_TEAMS),
        key=lambda idx: (-points[idx], random.random())
    )

    # Assign positions (1..36)
    position = {team: rank + 1 for rank, team in enumerate(ranking)}

    # Update statistics
    for team in range(N_TEAMS):
        pts = points[team]
        pos = position[team]
        data = results_per_points[pts]

        data["count"] += 1
        if pos <= 8:
            data["top8"] += 1
        if pos <= 16:
            data["top16"] += 1
        if pos <= 24:
            data["top24"] += 1
        data["positions"][pos] += 1

# ----- Summary output -----

print(
    "Pts | Samples |"
    "  P(top8) | Top8%  |"
    " P(top16) | Top16% |"
    " P(top24) | Top24% |"
    " Avg Pos"
)
print("-" * 90)

for pts in range(MAX_POINTS + 1):
    if pts not in results_per_points:
        continue

    data = results_per_points[pts]
    count = data["count"]
    if count == 0:
        continue

    top8_prob = data["top8"] / count
    top16_prob = data["top16"] / count
    top24_prob = data["top24"] / count

    top8_pct = top8_prob * 100
    top16_pct = top16_prob * 100
    top24_pct = top24_prob * 100

    total_pos = sum(pos * freq for pos, freq in data["positions"].items())
    avg_pos = total_pos / count

    print(
        f"{pts:>3} | {count:7d} |"
        f"  {top8_prob:6.3f} | {top8_pct:6.1f}% |"
        f"  {top16_prob:6.3f} | {top16_pct:6.1f}% |"
        f"  {top24_prob:6.3f} | {top24_pct:6.1f}% |"
        f"  {avg_pos:7.2f}"
    )
