#date: 2024-06-14T16:55:04Z
#url: https://api.github.com/gists/ff23847a7cc2cae4a3b9ce08827ddd0f
#owner: https://api.github.com/users/sylvainSUPINTERNET

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

def is_valid_combination(combination, constraints):
    for var, val in combination.items():
        if var in constraints and val in constraints[var]:
            for next_var, next_vals in constraints[var][val].items():
                if next_var in combination and combination[next_var] not in next_vals:
                    return False
    return True

def explore_combination(combination, constraints, explored):
    queue = deque([combination])
    local_combinations = []

    while queue:
        combination = queue.popleft()
        comb_key = tuple(sorted(combination.items()))

        if comb_key in explored:
            continue
        explored.add(comb_key)

        is_complete = all(var in combination for var in constraints.keys())
        if is_complete:
            local_combinations.append(combination)
            continue

        for var, val in combination.items():
            if var in constraints and val in constraints[var]:
                for next_var, next_vals in constraints[var][val].items():
                    for next_val in next_vals:
                        if next_var not in combination:
                            new_combination = combination.copy()
                            new_combination[next_var] = next_val
                            if is_valid_combination(new_combination, constraints):
                                queue.append(new_combination)

    return local_combinations

def generate_combinations(start_variable, start_values, constraints):
    combinations = []
    explored = set()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(explore_combination, {start_variable: value}, constraints, explored) for value in start_values]

        for future in as_completed(futures):
            combinations.extend(future.result())

    return combinations

# Contraintes mises à jour
constraints = {
    'D6E': {
        '00': {'REG': ['AR', 'AQ', 'AP', 'AM']},
        '01': {'REG': ['AR', 'AP']}
    },
    'REG': {
        'AR': {
            'D6E': ['00', '01'],
            'B0M': ['M0', 'M6', 'P0', 'M5'],
            'B0N': ['LD', 'F4', 'EQ', '9V', 'SM', 'WP', 'VH']
        },
        'AQ': {
            'D6E': ['00'],
            'B0M': ['M0', 'M6', 'P0', 'M5'],
            'B0N': ['LD', 'F4', 'EQ', '9V', 'SM', 'WP', 'VH', 'VL', '2T']
        },
        'AP': {
            'D6E': ['00', '01'],
            'B0M': ['M0', 'M6', 'P0', 'M5'],
            'B0N': ['LD', 'F4', 'EQ', '9V', 'SM', 'WP', 'VH']
        },
        'AM': {
            'D6E': ['00'],
            'B0M': ['M0', 'M6', 'P0', 'M5'],
            'B0N': ['F4', '9V', 'SM', 'WP', 'VH', 'VL', '2T']
        }
    },
    'B0M': {
        'M0': {
            'REG': ['AR', 'AQ', 'AP', 'AM'],
            'B0N': ['LD', 'F4', 'EQ', '9V', 'VL', '2T']
        },
        'M6': {
            'REG': ['AR', 'AQ', 'AP', 'AM'],
            'B0N': ['SM']
        },
        'P0': {
            'REG': ['AR', 'AQ', 'AP', 'AM'],
            'B0N': ['WP']
        },
        'M5': {
            'REG': ['AR', 'AQ', 'AP', 'AM'],
            'B0N': ['VH']
        }
    },
    'B0N': {
        'LD': {
            'REG': ['AR', 'AQ', 'AP']
        },
        'F4': {
            'REG': ['AR', 'AQ', 'AP', 'AM']
        },
        'EQ': {
            'REG': ['AR', 'AQ', 'AP']
        },
        '9V': {
            'REG': ['AR', 'AQ', 'AP', 'AM']
        },
        'SM': {
            'REG': ['AR', 'AQ', 'AP', 'AM']
        },
        'WP': {
            'REG': ['AR', 'AQ', 'AP', 'AM']
        },
        'VL': {
            'REG': ['AQ', 'AM']
        },
        'VH': {
            'REG': ['AR', 'AQ', 'AP', 'AM']
        },
        '2T': {
            'REG': ['AQ', 'AM']
        }
    }
}

# Début avec REG = AR, AQ, AP, AM
start_variable = 'B0N'
start_values = ['LD']
combinations = generate_combinations(start_variable, start_values, constraints)

# Afficher les combinaisons possibles
print("Generated combinations:")
for combination in combinations:
    print(combination)