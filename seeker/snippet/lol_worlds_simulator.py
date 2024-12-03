#date: 2024-12-03T16:52:15Z
#url: https://api.github.com/gists/6dbf7a20cfedd79134067a2ab22bead8
#owner: https://api.github.com/users/aureliony

import multiprocessing
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

teams = [
    'HLE', 'GEN', 'T1', 'DK',
    'BLG', 'TES', 'LNG', 'WBG',
    'G2', 'FNC',
    'FLY', 'TL',
    'MDK', 'GAM',
    'PNG', 'PSG'
]

elo_ratings_default = {
    "GEN": 1663, "BLG": 1602, "HLE": 1572, "TES": 1508, "G2": 1478, "T1": 1467,
    "DK": 1445, "LNG": 1407, "WBG": 1391, "FNC": 1372, "TL": 1362, "FLY": 1342,
    "PSG": 1323, "MDK": 1302, "GAM": 1254, "PNG": 1183
}
elo_ratings_default_with_g2_penalty = {
    "GEN": 1663, "BLG": 1602, "HLE": 1572, "TES": 1508, "G2": 1391, "T1": 1467,
    "DK": 1445, "LNG": 1407, "WBG": 1391, "FNC": 1372, "TL": 1362, "FLY": 1342,
    "PSG": 1323, "MDK": 1302, "GAM": 1254, "PNG": 1183
}
elo_ratings = {
    "GEN": 1663, "BLG": 1602, "HLE": 1572, "TES": 1508, "G2": 1391, "T1": 1467,
    "DK": 1445, "LNG": 1407, "WBG": 1391, "FNC": 1122, "TL": 1112, "FLY": 1092,
    "PSG": 1073, "MDK": 1052, "GAM": 1104, "PNG": 933
}
# elo_ratings = elo_ratings_default_with_g2_penalty

def elo_win_probability(team1, team2):
    R1 = elo_ratings[team1]
    R2 = elo_ratings[team2]
    prob_team1_wins = 1 / (1 + 10 ** ((R2 - R1) / 400))
    return prob_team1_wins


probs = {
    ('HLE', 'GEN') : 0.47,
    ('HLE', 'T1') : 0.66,
    ('HLE', 'DK') : 0.66,
    ('HLE', 'BLG') : 0.45,
    ('HLE', 'TES') : 0.5,
    ('HLE', 'LNG') : 0.7,
    ('HLE', 'WBG') : 0.7,
    ('HLE', 'G2') : 0.8,
    ('HLE', 'FNC') : 1.0,
    ('HLE', 'FLY') : 1.0,
    ('HLE', 'TL') : 1.0,
    ('HLE', 'MDK') : 1.0,
    ('HLE', 'GAM') : 1.0,
    ('HLE', 'PNG') : 1.0,
    ('HLE', 'PSG') : 0.95,

    ('GEN', 'T1') : 0.77,
    ('GEN', 'DK') : 0.75,
    ('GEN', 'BLG') : 0.5,
    ('GEN', 'TES') : 0.6,
    ('GEN', 'LNG') : 0.66,
    ('GEN', 'WBG') : 0.7,
    ('GEN', 'G2') : 0.8,
    ('GEN', 'FNC') : 1.0,
    ('GEN', 'FLY') : 1.0,
    ('GEN', 'TL') : 1.0,
    ('GEN', 'MDK') : 1.0,
    ('GEN', 'GAM') : 1.0,
    ('GEN', 'PNG') : 1.0,
    ('GEN', 'PSG') : 0.95,

    ('T1', 'DK') : 0.6,
    ('T1', 'BLG') : 0.5,
    ('T1', 'TES') : 0.45,
    ('T1', 'LNG') : 0.66,
    ('T1', 'WBG') : 0.7,
    ('T1', 'G2') : 0.75,
    ('T1', 'FNC') : 1.0,
    ('T1', 'FLY') : 1.0,
    ('T1', 'TL') : 1.0,
    ('T1', 'MDK') : 1.0,
    ('T1', 'GAM') : 1.0,
    ('T1', 'PNG') : 1.0,
    ('T1', 'PSG') : 0.9,

    ('DK', 'BLG') : 0.3,
    ('DK', 'TES') : 0.3,
    ('DK', 'LNG') : 0.3,
    ('DK', 'WBG') : 0.5,
    ('DK', 'G2')  : 0.5,
    ('DK', 'FNC') : 1.0,
    ('DK', 'FLY') : 1.0,
    ('DK', 'TL')  : 1.0,
    ('DK', 'MDK') : 1.0,
    ('DK', 'GAM') : 1.0,
    ('DK', 'PNG') : 1.0,
    ('DK', 'PSG') : 0.9,

    ('BLG', 'TES') : 0.63,
    ('BLG', 'LNG') : 0.75,
    ('BLG', 'WBG') : 0.77,
    ('BLG', 'G2') : 0.85,
    ('BLG', 'FNC') : 1.0,
    ('BLG', 'FLY') : 1.0,
    ('BLG', 'TL') : 1.0,
    ('BLG', 'MDK') : 1.0,
    ('BLG', 'GAM') : 1.0,
    ('BLG', 'PNG') : 1.0,
    ('BLG', 'PSG') : 0.8,

    ('TES', 'LNG') : 0.66,
    ('TES', 'WBG') : 0.66,
    ('TES', 'G2') : 0.75,
    ('TES', 'FNC') : 1.0,
    ('TES', 'FLY') : 1.0,
    ('TES', 'TL') : 1.0,
    ('TES', 'MDK') : 1.0,
    ('TES', 'GAM') : 1.0,
    ('TES', 'PNG') : 1.0,
    ('TES', 'PSG') : 0.8,

    ('LNG', 'WBG') : 0.52,
    ('LNG', 'G2') : 0.55,
    ('LNG', 'FNC') : 1.0,
    ('LNG', 'FLY') : 1.0,
    ('LNG', 'TL') : 1.0,
    ('LNG', 'MDK') : 1.0,
    ('LNG', 'GAM') : 1.0,
    ('LNG', 'PNG') : 1.0,
    ('LNG', 'PSG') : 0.8,

    ('WBG', 'G2') : 0.42,
    ('WBG', 'FNC') : 0.66,
    ('WBG', 'FLY') : 0.5,
    ('WBG', 'TL') : 0.75,
    ('WBG', 'MDK') : 1.0,
    ('WBG', 'GAM') : 1.0,
    ('WBG', 'PNG') : 1.0,
    ('WBG', 'PSG') : 0.8,

    ('G2', 'FNC') : 0.75,
    ('G2', 'FLY') : 0.66,
    ('G2', 'TL') : 0.75,
    ('G2', 'MDK') : 0.9,
    ('G2', 'GAM') : 1.0,
    ('G2', 'PNG') : 1.0,
    ('G2', 'PSG') : 0.7,

    ('FNC', 'FLY') : 0.5,
    ('FNC', 'TL') : 0.5,
    ('FNC', 'MDK') : 0.62,
    ('FNC', 'GAM') : 0.8,
    ('FNC', 'PNG') : 1.0,
    ('FNC', 'PSG') : 0.58,

    ('FLY', 'TL') : 0.5,
    ('FLY', 'MDK') : 0.5,
    ('FLY', 'GAM') : 0.7,
    ('FLY', 'PNG') : 1.0,
    ('FLY', 'PSG') : 0.5,

    ('TL', 'MDK') : 0.5,
    ('TL', 'GAM') : 0.7,
    ('TL', 'PNG') : 0.9,
    ('TL', 'PSG') : 0.5,

    ('MDK', 'GAM') : 0.8,
    ('MDK', 'PNG') : 1.0,
    ('MDK', 'PSG') : 0.6,

    ('GAM', 'PNG') : 0.5,
    ('GAM', 'PSG') : 0.2,

    ('PNG', 'PSG') : 0.0,
}
# probs = {}
# for i in range(len(teams)-1):
#     for j in range(i+1, len(teams)):
#         team = teams[i]
#         team2 = teams[j]
#         probs[(team, team2)] = elo_win_probability(team, team2)
# for k,v in probs.items():
#     print(k, ':', v, end=',\n')
# exit()

for (team1, team2), prob in list(probs.items()):
    probs[(team2, team1)] = 1.0 - prob

def win_probability(team1, team2):
    return probs[(team1, team2)]

# print(win_probability('HLE', 'G2'))
# exit()

# Function to simulate a match considering Elo ratings
def simulate_match(team1, team2):
    prob_team1_wins = win_probability(team1, team2)
    # Simulate the match based on probabilities
    if random.random() <= prob_team1_wins:
        return team1, team2
    else:
        return team2, team1


# Function to create matchups for a round based on win-loss records
def create_matchups(team_records, done_matchups):
    grouped_teams = defaultdict(list)
    for team, record in team_records.items():
        if record['wins'] < 3 and record['losses'] < 3:
            key = (record['wins'], record['losses'])
            grouped_teams[key].append(team)

    matchups = []
    for record_group, teams in grouped_teams.items():
        assert len(teams) % 2 == 0, \
            "No. of teams in the pool is odd"

        while True:
            random.shuffle(teams)
            test_matchups = [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]
            if not any(matchups in done_matchups for matchups in test_matchups):
                matchups.extend(test_matchups)
                break

    return matchups


# Function to simulate a round of matches
def simulate_round(matchups, team_records):
    for match in matchups:
        winner, loser = simulate_match(match[0], match[1])
        team_records[winner]['wins'] += 1
        team_records[loser]['losses'] += 1


# Function to check if the tournament should continue
def tournament_active(team_records):
    return any(record['wins'] < 3 and record['losses'] < 3 for record in team_records.values())


# Function to run simulations
def run_simulations(n):
    outcomes = Counter()

    for _ in range(n):
        team_records = {
            "HLE": {"wins": 3, "losses": 1},
            "GEN": {"wins": 3, "losses": 0},
            "T1":  {"wins": 3, "losses": 1},
            "DK":  {"wins": 2, "losses": 2},

            "BLG": {"wins": 2, "losses": 2},
            "TES": {"wins": 3, "losses": 1},
            "LNG": {"wins": 3, "losses": 0},
            "WBG": {"wins": 2, "losses": 2},

            "G2":  {"wins": 2, "losses": 2},
            "FNC": {"wins": 1, "losses": 3},
            "FLY": {"wins": 2, "losses": 2},
            "TL":  {"wins": 2, "losses": 2},

            "MDK": {"wins": 0, "losses": 3},
            "GAM": {"wins": 1, "losses": 3},
            "PNG": {"wins": 0, "losses": 3},
            "PSG": {"wins": 1, "losses": 3},
        }

        done_matchups = [
            ('MDK', 'BLG'),
            ('TES', 'T1'),
            ('GEN', 'WBG'),
            ('FNC', 'DK'),
            ('TL', 'LNG'),
            ('PSG', 'HLE'),
            ('FLY', 'GAM'),
            ('PNG', 'G2'),

            ('G2', 'HLE'),
            ('DK', 'FLY'),
            ('GEN', 'TES'),
            ('BLG', 'LNG'),
            ('T1', 'PNG'),
            ('PSG', 'MDK'),
            ('WBG', 'TL'),
            ('FNC', 'GAM'),

            ('DK', 'LNG'),
            ('HLE', 'GEN'),
            ('TES', 'FNC'),
            ('BLG', 'T1'),
            ('PSG', 'FLY'),
            ('WBG', 'G2'),
            ('MDK', 'GAM'),
            ('PNG', 'TL'),

            ('DK', 'TES'),
            ('HLE', 'FLY'),
            ('G2', 'T1'),
            ('BLG', 'PSG'),
            ('WBG', 'FNC'),
            ('TL', 'GAM'),
        ]

        current_matchups = [
            ('BLG', 'G2'),
            ('TL', 'FLY'),
            ('DK', 'WBG'),
        ]

        # current_matchups = done_matchups
        # done_matchups = []
        # for k in team_records:
        #     team_records[k]['wins'] = 0
        #     team_records[k]['losses'] = 0

        # Simulate the current round
        simulate_round(current_matchups, team_records)
        done_matchups.extend(current_matchups)
        # Continue playing rounds until each team has 3 wins or 3 losses
        while tournament_active(team_records):
            current_matchups = create_matchups(team_records, done_matchups)
            simulate_round(current_matchups, team_records)
            done_matchups.extend(current_matchups)

        # Count teams that achieved 3 wins
        teams_that_got_out = []
        for team, record in team_records.items():
            assert 0 <= record['wins'] <= 3
            assert 0 <= record['losses'] <= 3
            assert record['wins'] == 3 or record['losses'] == 3
            if record['wins'] == 3:
                teams_that_got_out.append(team)

        teams_that_got_out.sort(key=lambda team: teams.index(team))
        outcomes[tuple(teams_that_got_out)] += 1

    return outcomes


def plot_histogram(data: dict, top_k=20):
    # Extracting keys (items) and values (counts)
    items = [(', '.join(outcome), count) for outcome, count in data.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    outcomes, counts = zip(*items)
    total = sum(counts)
    probs = [count / total for count in counts]
    print(sum(probs[:top_k]), flush=True)

    # Plotting the bar chart
    plt.figure(figsize=(top_k*0.4, 8))
    plt.bar(outcomes[:top_k], probs[:top_k], color='skyblue')

    # Adding labels and title
    plt.xlabel('Swiss outcome')
    plt.ylabel('Probability')

    # Rotating the x labels for better readability
    plt.xticks(rotation=90)

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the simulations and print the probabilities
    num_simulations_per_worker = 10000
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    num_simulations = num_simulations_per_worker * num_workers
    numbers = [num_simulations_per_worker] * num_workers

    with multiprocessing.Pool(processes=num_workers) as pool:
        list_of_outcomes = pool.map(run_simulations, numbers)

    outcomes = sum(list_of_outcomes, start=Counter())

    probabilities = Counter()
    for team in teams:
        probabilities[team] = 0
    for teams, count in outcomes.items():
        for team in teams:
            probabilities[team] += count
    for team in probabilities:
        probabilities[team] /= num_simulations

    # Display the probabilities
    print(f"Probability of each team making it out ({num_simulations} simulations):")
    for team, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {prob:.2%}")
    print(flush=True)

    plot_histogram(outcomes)
