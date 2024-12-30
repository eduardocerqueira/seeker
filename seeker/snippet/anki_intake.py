#date: 2024-12-30T17:02:50Z
#url: https://api.github.com/gists/fe54cc9dbd6d5f100f6df69774832bd4
#owner: https://api.github.com/users/tnguyen21

import math

p = 0.9 # success rate
np = 1 - p
new_cards = 40
ease = 2.5
max_days = 365

queue = [{"base_days": 5, "factor": 1}]
total = 0

while queue:
    current = queue.pop(0)
    base_days = current["base_days"]
    factor = current["factor"]

    if base_days > max_days or factor < 0.0001:
        continue

    total += factor

    queue.append({"base_days": base_days * ease, "factor": factor * p})
    queue.append({"base_days": max(5, base_days / ease), "factor": factor * np})

print(f"total cards review:   {math.floor(total * new_cards)} cards/day")
print(f"total cards review:   {new_cards * max_days} cards/yr")
print(f"daily time (6s/card): {math.ceil(total * new_cards * 6)} sec")
print(f"daily time (6s/card): {math.ceil(total * new_cards * 6 / 60)} min")
