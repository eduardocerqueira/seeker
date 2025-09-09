#date: 2025-09-09T17:13:05Z
#url: https://api.github.com/gists/f6883a5e9135be7db344af93b5c49f7b
#owner: https://api.github.com/users/FrankSpooren

#!/usr/bin/env python3
# Verwijder oude semantic_search, behoud alleen nieuwe

with open('main.py', 'r') as f:
    lines = f.readlines()

# Zoek waar oude en nieuwe semantic_search staan
old_start = -1
new_start = -1

for i, line in enumerate(lines):
    if 'def semantic_search' in line:
        if old_start == -1:
            old_start = i
        else:
            new_start = i

print(f"Oude semantic_search op regel: {old_start}")
print(f"Nieuwe semantic_search op regel: {new_start}")

# Verwijder oude functie (van old_start tot new_start)
if old_start >= 0 and new_start > old_start:
    # Behoud alles voor oude functie en vanaf nieuwe functie
    new_lines = lines[:old_start] + lines[new_start:]
    
    with open('main_fixed_final.py', 'w') as f:
        f.writelines(new_lines)
    
    print("Created main_fixed_final.py zonder duplicate")