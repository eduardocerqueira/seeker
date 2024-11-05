#date: 2024-11-05T17:13:31Z
#url: https://api.github.com/gists/190ee46fbc57014f02d23b03cd92a93f
#owner: https://api.github.com/users/dmytro-audibene

"""
Campaign Total Cost Calculator (Python)

Problem Statement:
You are working on a marketing analytics tool. Your task is to implement a function called 'calculate_total_costs' that calculates the total cost of all active marketing campaigns for each day in the range of the campaigns.

Each campaign is represented by a tuple (start_day, end_day, daily_cost).
Days are represented as integers starting from 0.
The function should return a list where the index represents the day and the value represents the total cost of all active campaigns on that day.

Function Signature:
def calculate_total_costs(campaigns: List[Tuple[int, int, float]]) -> List[float]:

Example:
Input: 
campaigns = [(0, 5, 100.0), (2, 7, 150.0), (4, 9, 200.0)]

Output: [100.0, 100.0, 250.0, 250.0, 450.0, 450.0, 350.0, 350.0, 200.0, 200.0]

Explanation:
Day 0: 100.0 (first campaign starts)
Day 1: 100.0 (only first campaign active)
Day 2: 250.0 (second campaign starts, 100 + 150)
Day 3: 250.0 (first and second campaigns active)
Day 4: 450.0 (third campaign starts, 100 + 150 + 200)
Day 5: 450.0 (all three campaigns active)
Day 6: 350.0 (first campaign ends, 150 + 200)
Day 7: 350.0 (second and third campaigns active)
Day 8: 200.0 (second campaign ends, only third active)
Day 9: 200.0 (third campaign ends)

Constraints:
- 1 <= len(campaigns) <= 10^5
- 0 <= start_day < end_day <= 10^5
- 0.0 <= daily_cost <= 10000.0

Your task is to implement the 'calculate_total_costs' function efficiently.
"""

from typing import List, Tuple

def calculate_total_costs(campaigns: List[Tuple[int, int, float]]) -> List[float]:
    # Your implementation here
    return []

# Test cases
test_cases = [
    [(0, 5, 100.0), (2, 7, 150.0), (4, 9, 200.0)],
    [(0, 365, 50.0), (30, 90, 75.0), (91, 180, 100.0)],
    [(0, 30, 200.0), (31, 60, 250.0), (61, 90, 300.0)]
]

for i, case in enumerate(test_cases, 1):
    result = calculate_total_costs(case)
    print(f"Test case {i}: {result[:10]}{' ...' if len(result) > 10 else ''}")

# Expected outputs (showing first 10 days for brevity):
# Test case 1: [100.0, 100.0, 250.0, 250.0, 450.0, 450.0, 350.0, 350.0, 200.0, 200.0]
# Test case 2: [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, ...]
# Test case 3: [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, ...]