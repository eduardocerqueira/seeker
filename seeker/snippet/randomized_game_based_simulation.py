#date: 2025-12-10T17:14:20Z
#url: https://api.github.com/gists/497f55c09f33b24e6366006b7fadb81e
#owner: https://api.github.com/users/realKfiros

import random
from collections import Counter, defaultdict

# ----- Configuration -----

N_TEAMS = 36          # number of teams
ROUNDS = 8            # each team plays 8 games
N_SIM = 100_000       # number of simulated seasons

# Random match result:
# 1/3 win for team A (3 points)
# 1/3 draw (1–1)
# 1/3 win for team B (3 points)
def simulate_match():
    r = random.random()
    if r < 1/3:
        return 3, 0
    elif r < 2/3:
        return 1, 1
    else:
        return 0, 3

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

random.seed(42)  # for reproducibility

# ----- Simulation loop -----

for sim in range(N_SIM):
    points = [0] * N_TEAMS

    # Play 8 rounds
    for _ in range(ROUNDS):
        teams = list(range(N_TEAMS))
        random.shuffle(teams)

        # Each pair plays a match
        for i in range(0, N_TEAMS, 2):
            t1 = teams[i]
            t2 = teams[i + 1]
            p1, p2 = simulate_match()
            points[t1] += p1
            points[t2] += p2

    # Ranking: sort by points, then random tiebreaker
    ranking = sorted(
        range(N_TEAMS),
        key=lambda idx: (-points[idx], random.random())
    )

    # Assign positions
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

max_points = ROUNDS * 3  # 8 * 3 = 24

print(
    "Pts | Samples |"
    "  P(top8) | Top8%  |"
    " P(top16) | Top16% |"
    " P(top24) | Top24% |"
    " Avg Pos"
)
print("-" * 90)

for pts in range(max_points + 1):
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
