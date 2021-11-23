#date: 2021-11-23T16:59:31Z
#url: https://api.github.com/gists/6577e3391977971a7830c87b406fec28
#owner: https://api.github.com/users/DylanBaker

import itertools
import pandas as pd

TEAMS = [
    "Toria",
    "Dylan",
    "Tom",
    "Adam",
    "Sammie",
    "Will",
    "Liv",
    "Matt",
    "Emily",
    "John",
]

SCORES = {
    "1": {
        "Adam": 103.4,
        "Dylan": 87.32,
        "Emily": 130.4,
        "John": 127.88,
        "Liv": 131.78,
        "Matt": 102.82,
        "Sammie": 165.18,
        "Tom": 108.96,
        "Toria": 86.7,
        "Will": 118.26,
    },
    "2": {
        "Adam": 101.82,
        "Dylan": 133,
        "Emily": 100.74,
        "John": 97.96,
        "Liv": 75.32,
        "Matt": 111.18,
        "Sammie": 108.72,
        "Tom": 156.42,
        "Toria": 149.36,
        "Will": 112.2,
    },
    "3": {
        "Adam": 123.92,
        "Dylan": 121.14,
        "Emily": 75.94,
        "John": 113.12,
        "Liv": 106.44,
        "Matt": 107.62,
        "Sammie": 96.2,
        "Tom": 128.52,
        "Toria": 124.62,
        "Will": 102.34,
    },
    "4": {
        "Adam": 130.2,
        "Dylan": 92.12,
        "Emily": 81.76,
        "John": 90.92,
        "Liv": 135.18,
        "Matt": 128.02,
        "Sammie": 90.52,
        "Tom": 119.86,
        "Toria": 115.72,
        "Will": 125.32,
    },
    "5": {
        "Adam": 81,
        "Dylan": 159.48,
        "Emily": 116.54,
        "John": 109.4,
        "Liv": 147.52,
        "Matt": 120.18,
        "Sammie": 116.38,
        "Tom": 124.88,
        "Toria": 143.4,
        "Will": 103.66,
    },
    "6": {
        "Adam": 115.54,
        "Dylan": 84.38,
        "Emily": 87.58,
        "John": 108.62,
        "Liv": 147.3,
        "Matt": 115.9,
        "Sammie": 105.78,
        "Tom": 150.74,
        "Toria": 110.02,
        "Will": 134.06,
    },
    "7": {
        "Adam": 131.36,
        "Dylan": 82.98,
        "Emily": 115.74,
        "John": 114.68,
        "Liv": 89.14,
        "Matt": 107.94,
        "Sammie": 146.64,
        "Tom": 137.22,
        "Toria": 103.14,
        "Will": 96.44,
    },
    "8": {
        "Adam": 100.5,
        "Dylan": 94.66,
        "Emily": 108.9,
        "John": 113.76,
        "Liv": 132.92,
        "Matt": 104.96,
        "Sammie": 84.7,
        "Tom": 127.12,
        "Toria": 136.66,
        "Will": 63.36,
    },
    "9": {
        "Adam": 92.86,
        "Dylan": 98.74,
        "Emily": 129.92,
        "John": 91.28,
        "Liv": 106.44,
        "Matt": 99.38,
        "Sammie": 90.14,
        "Tom": 116.34,
        "Toria": 85.66,
        "Will": 83.7,
    },
    "10": {
        "Adam": 82.58,
        "Dylan": 123.32,
        "Emily": 77.4,
        "John": 81.8,
        "Liv": 104.7,
        "Matt": 103.54,
        "Sammie": 107.74,
        "Tom": 119.44,
        "Toria": 110.94,
        "Will": 138.08,
    },
}


def generate_permutations():
    print("Generating permutations...")
    permutations = list(itertools.permutations(TEAMS))
    print("Done!")
    return permutations


def generate_schedule(permutation, number_of_weeks):
    schedule = {}
    for week in range(1, number_of_weeks + 1):
        schedule[week] = []
        schedule[week].append([permutation[0], permutation[9]])
        schedule[week].append([permutation[1], permutation[8]])
        schedule[week].append([permutation[2], permutation[7]])
        schedule[week].append([permutation[3], permutation[6]])
        schedule[week].append([permutation[4], permutation[5]])
        permutation.append(permutation.pop(1))
    return schedule


def generate_all_schedules(number_of_weeks):
    print("Generating all schedules...")
    all_schedules = []
    for schedule_number, permutation in enumerate(generate_permutations()):
        if schedule_number % 10000 == 0:
            print(f"schedule: {schedule_number}")
        all_schedules.append(generate_schedule(list(permutation), number_of_weeks))
    print("Done!")
    return all_schedules


def calculate_result(team_one, team_two, week):
    team_one_sccore = SCORES[str(week)][team_one]
    team_two_score = SCORES[str(week)][team_two]
    if team_one_sccore > team_two_score:
        return team_one
    else:
        return team_two


def generate_start_table():
    table = {
        "Dylan": {"wins": 0, "points": 1195.34},
        "Tom": {"wins": 0, "points": 1435.60},
        "Adam": {"wins": 0, "points": 1166.58},
        "Toria": {"wins": 0, "points": 1239.38},
        "Sammie": {"wins": 0, "points": 1198.60},
        "Will": {"wins": 0, "points": 1201.94},
        "Liv": {"wins": 0, "points": 1324.62},
        "Matt": {"wins": 0, "points": 1202.98},
        "Emily": {"wins": 0, "points": 1133.10},
        "John": {"wins": 0, "points": 1154.26},
    }
    return table


def calculate_season(schedule, number_of_weeks):
    table = generate_start_table()
    for week in range(1, number_of_weeks + 1):
        for game in schedule[week]:
            team_one = game[0]
            team_two = game[1]
            winner = calculate_result(team_one, team_two, week)
            table[winner]["wins"] += 1
    return table


def calculate_final_positions(table, season_number):
    positions = [
        {"team": k, "wins": v["wins"], "points": v["points"], "season": season_number}
        for k, v in table.items()
    ]
    return positions


def run_simulation(number_of_weeks):
    results = []
    schedules = generate_all_schedules(number_of_weeks)
    for season_number, schedule in enumerate(schedules):
        if season_number % 10000 == 0:
            print(season_number)
        season = calculate_season(schedule, number_of_weeks)
        positions = calculate_final_positions(season, season_number)
        results += positions
    return results


def output_data(results):
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    number_of_weeks = 10
    results = run_simulation(number_of_weeks)
    output_data(results)
